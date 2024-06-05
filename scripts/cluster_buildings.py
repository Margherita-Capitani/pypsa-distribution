# -*- coding: utf-8 -*-
import json
import logging
import os
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
from _helpers_dist import (
    configure_logging,
    sets_path_to_root,
    two_2_three_digits_country,
)
from shapely.geometry import shape
from sklearn.cluster import KMeans

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def extract_points(input_file, output_file):
    # Load GeoJSON data from file

    with open(input_file, "r") as f:
        data = json.load(f)

    # Create new GeoJSON data with only 'id' and 'coordinates' properties
    new_data = {"type": "FeatureCollection", "features": []}
    for feature in data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            polygon = shape(feature["geometry"])
            centroid = polygon.centroid
            centroid_coordinates = [centroid.x, centroid.y]
            new_feature = {
                "type": "Feature",
                "properties": {
                    "id": feature["properties"]["id"],
                    "tags.building": feature["properties"]["tags.building"],
                },
                "geometry": {
                    "type": feature["geometry"]["type"],
                    "coordinates": feature["geometry"]["coordinates"],
                    "center_coordinates": centroid_coordinates,
                },
            }
            new_data["features"].append(new_feature)

    # Write new GeoJSON data to file
    with open(output_file, "w") as f:
        json.dump(new_data, f)


def get_central_points_geojson(input_filepath, output_filepath, n_clusters):
    # Load GeoJSON data from file
    with open(input_filepath, "r") as f:
        data = json.load(f)

    # Extract coordinates of all points
    points = []
    for feature in data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            points.append(feature["geometry"]["center_coordinates"])

    # Convert points to NumPy array for use with scikit-learn
    points = np.array(points)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)

    # Compute centroids of each cluster
    centroids = kmeans.cluster_centers_

    # Select most central point in each cluster
    central_points = []
    for i in range(kmeans.n_clusters):
        cluster_points = points[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        central_point_idx = np.argmin(distances)
        central_points.append(cluster_points[central_point_idx])

    # Create GeoJSON feature for each central point
    features = []
    for i, central_point in enumerate(central_points):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": central_point.tolist()},
            "properties": {"cluster": i},
        }
        features.append(feature)

    # Create GeoJSON object with all central point features
    geojson = {"type": "FeatureCollection", "features": features}

    # Write central point GeoJSON data to file
    with open(output_filepath, "w") as f:
        json.dump(geojson, f)


def get_central_points_geojson_with_buildings(
    input_filepath, output_filepath, n_clusters
):
    # Load GeoJSON data from file
    with open(input_filepath, "r") as f:
        data = json.load(f)

    # Extract coordinates of all points
    points = []
    for feature in data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            points.append(feature["geometry"]["center_coordinates"])

    # Convert points to NumPy array for use with scikit-learn
    points = np.array(points)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)

    # Compute centroids of each cluster
    centroids = kmeans.cluster_centers_

    # Select most central point in each cluster
    central_points = []
    for i in range(kmeans.n_clusters):
        cluster_points = points[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        central_point_idx = np.argmin(distances)
        central_points.append(cluster_points[central_point_idx])

    # Create GeoJSON feature for each central point
    features = []
    for i, central_point in enumerate(central_points):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": central_point.tolist()},
            "properties": {"cluster": i},
        }
        features.append(feature)

    # Create GeoJSON object with all central point features
    geojson = {"type": "FeatureCollection", "features": features}

    # Assign each building to its corresponding cluster based on k-means label
    for i, label in enumerate(kmeans.labels_):
        feature = data["features"][i]
        if feature["geometry"]["type"] == "Polygon":
            cluster_id = label
            if "buildings" not in geojson["features"][cluster_id]["properties"]:
                geojson["features"][cluster_id]["properties"]["buildings"] = []
            geojson["features"][cluster_id]["properties"]["buildings"].append(
                feature["properties"]["tags.building"]
            )

    # Write central point GeoJSON data with buildings to file
    with open(output_filepath, "w") as f:
        json.dump(geojson, f)
    print("c")


def get_number_type_buildings(input_filepath, output_filepath):

    with open(input_filepath, "r") as f:
        data = json.load(f)

    cluster_buildings_count = {}

    for feature in data["features"]:
        cluster = feature["properties"]["cluster"]
        buildings = feature["properties"]["buildings"]

        processed_buildings = []
        for building in buildings:
            if building is None:
                processed_buildings.append("yes")
            else:
                processed_buildings.append(building)

        if cluster not in cluster_buildings_count:
            cluster_buildings_count[cluster] = Counter()

        cluster_buildings_count[cluster].update(processed_buildings)

    count = pd.DataFrame(cluster_buildings_count)
    count.fillna(0, inplace=True)
    count.to_excel(output_filepath)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers_dist import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("cluster_buildings")
        sets_path_to_root("pypsa-distribution")

    configure_logging(snakemake)

    extract_points(
        snakemake.input["buildings_geojson"],
        snakemake.output["cleaned_buildings_geojson"],
    )

    get_central_points_geojson(
        snakemake.output["cleaned_buildings_geojson"],
        snakemake.output["clusters"],
        snakemake.config["buildings"]["n_clusters"],
    )

    get_central_points_geojson_with_buildings(
        snakemake.output["cleaned_buildings_geojson"],
        snakemake.output["clusters_with_buildings"],
        snakemake.config["buildings"]["n_clusters"],
    )

    get_number_type_buildings(
        snakemake.output["clusters_with_buildings"],
        snakemake.output["number_buildings_type"],
    )
