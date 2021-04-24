"""
 Continue on from here : https://docs.streamlit.io/en/stable/

Todo:
* Initiate Unsupervised Labeling
    ** Clean up UI.
        ** Allow UI option to specify minimum number of labels, per category 
        ** Update Active Labeled List with an index, only if that has not already been added to that category.
            ** Remove from any previous category. 
            ** Add to any other category.
        ** Once enough labels are collected, cache this information.
* Initiate Active Labeling
    ** DONE - On this page, split into train / test sets. 
    ** DONE - Prepare a test dataset for evaluation, and metrics.
    ** DONE - Fit an initial model with training data from Initial Learning.
    ** DONE - Perform Active Learning iteratively, AND 
        *** Fit model(s) at each step, plus
        *** Print / write the Confusion Matrices of the model at each step of training.
        *** The model did learn !!!!
    ** Update training data and test data gradually after application of each label.
        *** In main data structure, or otherwise.

* Once above is done, we should be ready for the demo.

"""
# Imports
import streamlit as st
from streamlit.report_thread import get_report_ctx
import SessionState

import numpy as np
from modAL.models import ActiveLearner

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from time import time, sleep

import pandas as pd

import warnings

# A variable to keep track of which product we are currently displaying
session_state = SessionState.get(
    image_number=0,
    active_learning_dataset_indices=dict(),
    active_learning_index_to_label_map=dict(),
)


# Constants
PAGE_TITLE = "Active Learning Dashboard"
ACTION_SELECT_IMAGE_TO_LABEL = "Select Image to Label"
ACTION_SKIP = "Skip"

RANDOM_SEED = 1234
STANDARD_SEPARATOR = "##########################################################################################"
NUM_K_MEANS_INITIALIZATIONS = 4
NUM_CLUSTER_SAMPLES_TO_REVIEW = 2

MINIMUM_SAMPLES_PER_LABEL_CATEGORY_TO_COLLECT = 2

# Initialize Important Libraries

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")


# Setup the page

st.set_page_config(page_title=PAGE_TITLE, layout="wide")


@st.cache
def load_dataset():
    with st.spinner("Loading Digits Dataset..."):
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        X_image = digits.images
        X = digits.images.reshape((n_samples, -1))
        y = digits.target

        # Filter to specific label categories ( to reduce the labeling effort )
        preferred_labels_to_include = [0, 1, 2]
        filtered_indices = list()
        for preferred_label in preferred_labels_to_include:
            indices_matching_label = list(np.where(y == preferred_label))[0]
            filtered_indices.extend(indices_matching_label)

        x_image_filtered = X_image[filtered_indices]
        x_filtered = X[filtered_indices]
        y_filtered = y[filtered_indices]
        n_samples_filtered = len(filtered_indices)

    return x_image_filtered, x_filtered, y_filtered, n_samples_filtered


@st.cache(allow_output_mutation=True)
def get_active_learner(X_train_initial, y_train_initial):
    learning_model = RandomForestClassifier(random_state=RANDOM_SEED)
    active_learner = ActiveLearner(
        estimator=learning_model, X_training=X_train_initial, y_training=y_train_initial
    )
    return active_learner


@st.cache
def execute_clustering(X, y):
    # Run k-Means Clustering on the data
    unique_labels = list(np.unique(y))
    num_unique_labels = len(unique_labels)

    kmeans = KMeans(
        init="k-means++",
        n_clusters=num_unique_labels,
        n_init=NUM_K_MEANS_INITIALIZATIONS,
        random_state=RANDOM_SEED,
    )

    with st.spinner("Running k-Means Clustering..."):
        sleep(1)
        start_time = time()
        clustering_pipeline = make_pipeline(StandardScaler(), kmeans).fit(X)
        fitted_model = clustering_pipeline[-1]
        end_time = time()
        fit_time = end_time - start_time

        print(
            f"Results - Fit Time - {fit_time} || Inertia : {fitted_model.inertia_} || Data Size : {n_samples} "
        )

    return fitted_model, fit_time, unique_labels, clustering_pipeline


def inspect_cluster_quality(fitted_model):
    with st.beta_expander("Inspect Assigned Clusters"):
        clustered_labels = fitted_model.labels_
        unique_clustered_labels = np.unique(clustered_labels)

        clustered_labels_indices_dict = dict()

        for unique_clustered_label in unique_clustered_labels:
            print(STANDARD_SEPARATOR)
            print(
                f"Inspecting Cluster assignment for Cluster Label : {unique_clustered_label}"
            )
            indices_matching_label_tuple = np.where(
                clustered_labels == unique_clustered_label
            )
            indices_matching_label = indices_matching_label_tuple[0]
            sampled_indices_matching_label = indices_matching_label[
                0:NUM_CLUSTER_SAMPLES_TO_REVIEW
            ]
            print(len(sampled_indices_matching_label))

            clustered_labels_indices_dict[
                unique_clustered_label
            ] = indices_matching_label

            ground_truth_labels = [
                y[sampled_index_matching_label]
                for sampled_index_matching_label in sampled_indices_matching_label
            ]

            with st.beta_container():
                st.write(
                    f"Clustered Label : {unique_clustered_label} || Actual Ground Truths : {ground_truth_labels}"
                )

                for sampled_index_matching_label in sampled_indices_matching_label:
                    ground_truth = y[sampled_index_matching_label]
                    label_image_input = X_image[sampled_index_matching_label]
                    st.write(ground_truth)
                    st.image(label_image_input, width=100, clamp=True)

            print(f"Ground Truths of Clustered Values : {ground_truth_labels}")

    return clustered_labels_indices_dict


@st.cache
def get_data_for_active_learning(active_learning_index_to_label_map):
    # Load data from Cache
    X_image, X, y, n_samples = load_dataset()

    unique_labels = list(np.unique(y))

    active_X_indices = list(session_state.active_learning_index_to_label_map.keys())
    active_y = list(session_state.active_learning_index_to_label_map.values())

    # X_remaining is the remaining set of images which have not been labeled yet
    X_remaining = np.delete(X, active_X_indices, axis=0)
    X_image_remaining = np.delete(X_image, active_X_indices, axis=0)
    y_remaining = np.delete(y, active_X_indices, axis=0)

    # Separate into training and test datasets

    (
        active_x_indices_train,
        active_x_indices_test,
        active_y_train,
        active_y_test,
    ) = train_test_split(
        active_X_indices,
        active_y,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=active_y,
    )

    # Get distribution of Training Data

    (labels_training, labels_training_counts) = np.unique(
        active_y_train, return_counts=True
    )

    # Get distribution of Training Data

    (labels_testing, labels_testing_counts) = np.unique(
        active_y_test, return_counts=True
    )

    return (
        X,
        labels_training,
        labels_training_counts,
        labels_testing,
        labels_testing_counts,
        active_x_indices_train,
        active_x_indices_test,
        active_y_train,
        active_y_test,
        X_remaining,
        X_image_remaining,
        unique_labels,
        y_remaining,
    )


if __name__ == "__main__":

    session_id = get_report_ctx().session_id
    st.sidebar.write(f"Session ID : {session_id}")

    page = st.sidebar.selectbox(
        "Select a page:",
        (
            "Select a page",
            "Load dataset",
            "Execute Clustering",
            "Initial Labeling",
            "Active Learning",
        ),
    )

    if st.sidebar.button("Reset Collected Labels"):
        session_state.active_learning_dataset_indices.clear()
        session_state.active_learning_index_to_label_map.clear()

    if page == "Select a page":
        st.write("Select a page from the menu")

    elif page == "Load dataset":
        # Load the dataset
        X_image, X, y, n_samples = load_dataset()
        st.success(f"Completed loading Digits Dataset : {y.shape} ")

    elif page == "Execute Clustering":
        X_image, X, y, n_samples = load_dataset()
        # Execute Clustering
        fitted_model, fit_time, unique_labels, clustering_pipeline = execute_clustering(
            X, y
        )
        st.success(f"Completed running k-means Clustering in : {fit_time} seconds")

        # Briefly evaluate the quality of Clusters
        clustered_labels_indices_dict = inspect_cluster_quality(fitted_model)

    elif page == "Initial Labeling":

        X_image, X, y, n_samples = load_dataset()
        # Execute Clustering
        fitted_model, fit_time, unique_labels, clustering_pipeline = execute_clustering(
            X, y
        )

        # Briefly evaluate the quality of Clusters
        clustered_labels_indices_dict = inspect_cluster_quality(fitted_model)

        # Prepare a list of images for labeling, in sequential order of cluster assignments
        lists_of_clustered_indices = list(clustered_labels_indices_dict.values())

        element_wise_lists_of_clustered_indices = list(zip(*lists_of_clustered_indices))

        flat_element_wise_lists_of_clustered_indices = [
            item
            for sublist in element_wise_lists_of_clustered_indices
            for item in sublist
        ]

        last_page = len(flat_element_wise_lists_of_clustered_indices)

        prev_container, middle_container, next_container = st.beta_columns([1, 4, 1])

        if next_container.button("Next"):
            if session_state.image_number + 1 > last_page:
                session_state.image_number = 0
            else:
                session_state.image_number += 1

        if prev_container.button("Previous"):

            if session_state.image_number - 1 < 0:
                session_state.image_number = last_page
            else:
                session_state.image_number -= 1

        with middle_container:

            image_index = flat_element_wise_lists_of_clustered_indices[
                session_state.image_number
            ]
            ground_truth_label = y[image_index]

            st.write(
                f"Image Number: {session_state.image_number} || Image Index: {image_index} || Ground Truth: {ground_truth_label}"
            )

            placeholder = st.empty()

            selected_label = placeholder.selectbox(
                "Select a label for the image", [ACTION_SKIP] + unique_labels
            )

            if selected_label != ACTION_SKIP:

                existing_labeled_samples_in_active_dataset = (
                    session_state.active_learning_dataset_indices.get(
                        selected_label, list()
                    )
                )
                num_existing_labeled_samples_for_ground_truth = len(
                    existing_labeled_samples_in_active_dataset
                )
                existing_labeled_samples_in_active_dataset.append(image_index)
                session_state.active_learning_dataset_indices[
                    selected_label
                ] = existing_labeled_samples_in_active_dataset

                session_state.active_learning_index_to_label_map[
                    image_index
                ] = selected_label

                placeholder.empty()

                st.write(f"Thank you for selecting the label : {selected_label}")

            image_to_label = X_image[image_index]
            st.image(image_to_label, width=400, clamp=True)

            label_keys = session_state.active_learning_dataset_indices.keys()

            for (
                labelkey,
                list_of_indices,
            ) in session_state.active_learning_dataset_indices.items():
                st.write(f"Label - {labelkey}|| Count - {len(list_of_indices)}||")

    elif page == "Active Learning":

        # Split collected labeled dataset into training and testing data
        st.write(
            f"Number of Collected Ground Truths : {len(session_state.active_learning_index_to_label_map)}"
        )

        (
            X,
            labels_training,
            labels_training_counts,
            labels_testing,
            labels_testing_counts,
            active_x_indices_train,
            active_x_indices_test,
            active_y_train,
            active_y_test,
            X_remaining,
            X_image_remaining,
            unique_labels,
            y_remaining,
        ) = get_data_for_active_learning(
            session_state.active_learning_index_to_label_map
        )

        st.write(
            f"Training Data || Labels : {labels_training} || Counts : {labels_training_counts}"
        )

        # Print distribution of Testing Data

        st.write(
            f"Testing Data || Labels : {labels_testing} || Counts : {labels_testing_counts}"
        )

        with st.spinner("Started fitting model...."):

            X_train_initial = X[active_x_indices_train]
            y_train_initial = active_y_train

            X_test = X[active_x_indices_test]
            y_test = active_y_test

            active_learner = get_active_learner(X_train_initial, y_train_initial)

            # Evaluate the Model at this initial stage
            # y_test_predicted = active_learner.predict(X_test)

            # classification_report_dict = metrics.classification_report(
            #     y_test, y_test_predicted, output_dict=True
            # )
            # classification_report_df = pd.DataFrame(
            #     classification_report_dict
            # ).transpose()

            # st.write(f"Initial Model Performance Characteristics: ")
            # st.write(classification_report_df)

        query_index, query_instance = active_learner.query(X_remaining)
        st.write(
            f"Query Index : {query_index} || Label hint : {y_remaining[query_index]} || {type(y_remaining[query_index])}"
        )

        image_for_index = X_image_remaining[query_index][-1]

        st.image(image_for_index, width=400, clamp=True)

        placeholder = st.empty()

        selected_label = placeholder.selectbox(
            "Select a label for the image", [ACTION_SKIP] + unique_labels
        )

        if selected_label != ACTION_SKIP:
            active_learner.teach(query_instance, [selected_label])

            st.write(
                f"Teaching completed with : {query_index} || as label : {[selected_label]}"
            )

            st.write(f"Query instance : {query_instance}")

            placeholder.empty()

            selected_label = placeholder.selectbox(
                "Select a label for the image",
                [ACTION_SKIP] + unique_labels,
                key="lasso",
            )

        # Evaluate the Model at this initial stage
        y_test_predicted_latest = active_learner.predict(X_test)

        classification_report_dict_latest = metrics.classification_report(
            y_test, y_test_predicted_latest, output_dict=True
        )
        classification_report_df_latest = pd.DataFrame(
            classification_report_dict_latest
        ).transpose()

        st.write(f"Latest Model Performance Characteristics with : {query_index} ")
        st.write(classification_report_df_latest)
