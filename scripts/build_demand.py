# -*- coding: utf-8 -*-
"""
Estimates the population and the electric load of each microgrid.

Relevant Settings
-----------------

.. code:: yaml

    microgrids_list:
        microgridX: 
          lon_min:
          lon_max: 
          lat_min: 
          lat_max: 
    load:
        scaling_factor:

Inputs
------
- ``data/sample_profile.csv``: a load profile, which will be scaled through a scaling_factor to obtain the per person load

Outputs
-------
- ``resources/shapes/microgrid_shapes.geojson``: a geojson file of the shape of each microgrid,
- ``resources/masked_files/masked_file_{i+1}.tif``,
- ``resources/demand/microgrid_load_{i+1}.csv``: the electric load of the microgid,

Description
-----------
The rule :mod:`build_demand` contains functions that are used to create a shape file of the microgrid, to mask a raster with the shape file and to estimate 
the population. Then the population is multiplied for the per person load and the microgrid load is then obtained. The process applies to all the microgrids specified in config.yaml.
"""

import json
import logging
import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import rasterio
import rasterio.mask
import requests
from _helpers_dist import (
    configure_logging,
    sets_path_to_root,
    two_2_three_digits_country,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def get_WorldPop_data(
    country_code,
    year,
    update=False,
    out_logging=False,
    size_min=300,
):
    """
    Download tiff file for each country code using the standard method from worldpop datastore with 1kmx1km resolution.

    Parameters
    ----------
    country_code : str
        Two letter country codes of the downloaded files.
        Files downloaded from https://data.worldpop.org/ datasets WorldPop UN adjusted
    year : int
        Year of the data to download
    update : bool
        Update = true, forces re-download of files
    size_min : int
        Minimum size of each file to download
    Returns
    -------
    WorldPop_inputfile : str
        Path of the file
    """

    three_digits_code = two_2_three_digits_country(country_code)

    if out_logging:
        _logger.info("Get WorldPop datasets")

    if country_code == "XK":
        WorldPop_filename = f"srb_ppp_{year}_UNadj_constrained.tif"
        WorldPop_urls = [
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/SRB/{WorldPop_filename}",
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/SRB/{WorldPop_filename}",
        ]
    else:
        WorldPop_filename = (
            f"{three_digits_code.lower()}_ppp_{year}_UNadj_constrained.tif"
        )
        # Urls used to possibly download the file
        WorldPop_urls = [
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/{two_2_three_digits_country(country_code).upper()}/{WorldPop_filename}",
            f"https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/{two_2_three_digits_country(country_code).upper()}/{WorldPop_filename}",
        ]

    WorldPop_inputfile = os.path.join(
        os.getcwd(),
        "pypsa-earth",
        "data",
        "WorldPop",
        WorldPop_filename,
    )  # Input filepath tif

    if not os.path.exists(WorldPop_inputfile) or update is True:
        if out_logging:
            _logger.warning(
                f"{WorldPop_filename} does not exist, downloading to {WorldPop_inputfile}"
            )
        #  create data/osm directory
        os.makedirs(os.path.dirname(WorldPop_inputfile), exist_ok=True)

        loaded = False
        for WorldPop_url in WorldPop_urls:
            with requests.get(WorldPop_url, stream=True) as r:
                with open(WorldPop_inputfile, "wb") as f:
                    if float(r.headers["Content-length"]) > size_min:
                        shutil.copyfileobj(r.raw, f)
                        loaded = True
                        break
        if not loaded:
            _logger.error(f"Impossible to download {WorldPop_filename}")

    return WorldPop_inputfile, WorldPop_filename


# Estimate the total population of tghe microgrid
def estimate_microgrid_population(
    n, p, raster_path, shapes_path, sample_profile, output_file
):
    # Read the sample profile of electricity demand and extract the column corresponding to the electric load
    per_unit_load = pd.read_csv(sample_profile)["0"] / p

    # Dataframe of the load
    microgrid_load = pd.DataFrame()

    # Load the GeoJSON file with the shapes to mask the raster
    shapes = gpd.read_file(shapes_path)

    # Mask the raster with each shape and save each masked raster as a new file
    for i, shape in shapes.iterrows():
        with rasterio.open(raster_path) as src:
            # Mask the raster with the current shape
            masked, out_transform = rasterio.mask.mask(src, [shape.geometry], crop=True)
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": masked.shape[1],
                    "width": masked.shape[2],
                    "transform": out_transform,
                }
            )

        pop_microgrid = masked[masked >= 0].sum()

        col_name = "microgrid_1_bus_572666767"
        microgrid_load[col_name] = per_unit_load * pop_microgrid

    return pop_microgrid, microgrid_load


def calculate_load(
    n,
    p,
    raster_path,
    shapes_path,
    sample_profile,
    geojson_file,
    output_file,
    input_path,
):
    # Estimate the microgrid population and load using the existing function
    pop_microgrid, microgrid_load = estimate_microgrid_population(
        n, p, raster_path, shapes_path, sample_profile, output_file
    )

    buildings_csv = pd.read_csv(input_path)
    total_buildings = buildings_csv.values.sum()
    buildings_for_cluster = []
    for row in buildings_csv.itertuples(index=False, name=None):
        cluster_buildings = sum(row)
        buildings_for_cluster.append(cluster_buildings)
    cluster_info_df = pd.DataFrame(buildings_for_cluster, columns=["Buildings"])
    # Calculate the number population per building
    population_per_building = pop_microgrid / total_buildings
    cluster_info_df["population"] = (
        cluster_info_df["Buildings"] * population_per_building
    )

    # Calculate the per unit load
    per_unit_load = pd.read_csv(sample_profile)["0"] / p
    load_df = cluster_info_df["population"].apply(lambda x: x * per_unit_load)
    new_index_names = [f"bus_{i}" for i in range(len(load_df))]
    load_df = load_df.rename(index=dict(zip(load_df.index, new_index_names)))
    load_df = load_df.T
    # Remove the bus_9 column
    load_df = load_df.drop("bus_9", axis=1)
    load_df.insert(0, "snapshots", n.snapshots)
    load_df.set_index("snapshots", inplace=True)
    load_df.to_csv(output_file, index=True)

    # Return the DataFrame
    return load_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers_dist import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("build_demand")
        sets_path_to_root("pypsa-distribution")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.create_network)
    sample_profile = snakemake.input["sample_profile"]

    assert (
        len(snakemake.config["countries"]) == 1
    ), "Error: only a country shall be specified"

    worldpop_path, worldpop_flname = get_WorldPop_data(
        snakemake.config["countries"][
            0
        ],  # TODO: this needs fix to generalize the countries
        snakemake.config["build_shape_options"]["year"],
        False,
    )

    estimate_microgrid_population(
        n,
        snakemake.config["load"]["scaling_factor"],
        worldpop_path,
        snakemake.input["microgrid_shapes"],
        sample_profile,
        snakemake.output["electric_load"],
    )

    calculate_load(
        n,
        snakemake.config["load"]["scaling_factor"],
        worldpop_path,
        snakemake.input["microgrid_shapes"],
        sample_profile,
        snakemake.input["clusters_with_buildings"],
        snakemake.output["electric_load"],
        snakemake.input["building_csv"],
    )
