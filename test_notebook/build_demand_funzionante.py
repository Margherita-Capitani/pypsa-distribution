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


def calculate_load_ramp(
    input_file_buildings,
    n,
    p,
    raster_path,
    shapes_path,
    sample_profile,
    output_file,
    input_file_profile_tier1,
    input_file_profile_tier2,
    input_file_profile_tier3,
    input_file_profile_tier4,
    input_file_profile_tier5,
    output_path_csv,
    tier_percent,
):
    cleaned_buildings = gpd.read_file(input_file_buildings)
    house = cleaned_buildings[cleaned_buildings["tags_building"] == "house"]
    area_tot = house["area_m2"].sum()

    pop_microgrid, microgrid_load = estimate_microgrid_population(
        n, p, raster_path, shapes_path, sample_profile, output_file
    )
    density = pop_microgrid / area_tot

    grouped_buildings = cleaned_buildings.groupby("cluster_id")
    clusters = np.sort(cleaned_buildings["cluster_id"].unique())
    house_area_for_cluster = []
    for cluster in clusters:
        cluster_buildings = pd.DataFrame(grouped_buildings.get_group(cluster))
        house = cluster_buildings[cluster_buildings["tags_building"] == "house"]
        area_house = house["area_m2"].sum()
        house_area_for_cluster.append(area_house)

    population_df = pd.DataFrame()
    population_df["cluster"] = clusters
    population_df.set_index("cluster", inplace=True)
    population_df["house_area_for_cluster"] = house_area_for_cluster
    people_for_cluster = (population_df["house_area_for_cluster"] * density).round()
    population_df["people_for_cluster"] = people_for_cluster

    # tier_percent = [0.2, 0.2, 0.3, 0.2, 0.05, 0.05]
    people_for_cluster = population_df["people_for_cluster"]
    tier_pop_df = population_df["people_for_cluster"].apply(
        lambda x: pd.Series([x * y for y in tier_percent])
    )
    demand_tier_1 = pd.read_excel(input_file_profile_tier1)
    demand_tier_2 = pd.read_excel(input_file_profile_tier2)
    demand_tier_3 = pd.read_excel(input_file_profile_tier3)
    demand_tier_4 = pd.read_excel(input_file_profile_tier4)
    demand_tier_5 = pd.read_excel(input_file_profile_tier5)

    # Creazione di un DataFrame con tutti i tier e la domanda media oraria per ognuno
    mean_demand_tier_df = pd.DataFrame()
    demand_tiers = [
        demand_tier_1,
        demand_tier_2,
        demand_tier_3,
        demand_tier_4,
        demand_tier_5,
    ]

    for i, demand_tier in enumerate(demand_tiers, start=1):
        mean_column_name = f"tier_{i}"
        mean_demand_tier_df[mean_column_name] = demand_tier["mean"]
    mean_demand_tier_df.insert(0, "tier_0", np.zeros(len(mean_demand_tier_df)))

    hours_index = pd.date_range(
        start="00:00:00", periods=len(mean_demand_tier_df), freq="H", normalize=True
    )
    mean_demand_tier_df.index = hours_index.time

    date_range = pd.date_range(start="2013-01-01", end="2013-12-31", freq="D")
    yearly_mean_demand_tier_df = pd.concat(
        [mean_demand_tier_df] * len(date_range), ignore_index=True
    )
    date_time_index = pd.date_range(
        start="2013-01-01", end="2013-12-31 23:00:00", freq="H"
    )
    yearly_mean_demand_tier_df.index = date_time_index

    dataframes_dict = {}
    for index, row in tier_pop_df.iterrows():
        loads = []
        for col in range(tier_pop_df.shape[1]):
            n = row[col]
            tier = yearly_mean_demand_tier_df.iloc[:, col]
            loads_h = n * tier
            loads.append(loads_h)

        loads_df = pd.DataFrame(loads)
        loads_df = loads_df.T

        nome_dataframe = f"bus_{index}"
        dataframes_dict[nome_dataframe] = loads_df

    tot_loads_dict = {}
    for key in dataframes_dict:
        load = dataframes_dict[key]
        load_tot = load.sum(axis=1)

        total_loads_df = pd.DataFrame(load_tot, columns=[f"{key}"])
        tot_loads_dict[key] = total_loads_df

    tot_loads_df = pd.DataFrame()
    for key, cluster_load in tot_loads_dict.items():
        tot_loads_df = pd.concat([tot_loads_df, cluster_load], axis=1)
    tot_loads_df.to_csv(output_path_csv)

    print("fino a qui tutto bene")


def calculate_load_ramp_std(
    input_file_buildings,
    n,
    p,
    raster_path,
    shapes_path,
    sample_profile,
    output_file,
    input_file_profile_tier1,
    input_file_profile_tier2,
    input_file_profile_tier3,
    input_file_profile_tier4,
    input_file_profile_tier5,
    output_path_csv,
    tier_percent,
):
    cleaned_buildings = gpd.read_file(input_file_buildings)
    house = cleaned_buildings[cleaned_buildings["tags_building"] == "house"]
    area_tot = house["area_m2"].sum()

    pop_microgrid, microgrid_load = estimate_microgrid_population(
        n, p, raster_path, shapes_path, sample_profile, output_file
    )
    density = pop_microgrid / area_tot

    grouped_buildings = cleaned_buildings.groupby("cluster_id")
    clusters = np.sort(cleaned_buildings["cluster_id"].unique())
    house_area_for_cluster = []
    for cluster in clusters:
        cluster_buildings = pd.DataFrame(grouped_buildings.get_group(cluster))
        house = cluster_buildings[cluster_buildings["tags_building"] == "house"]
        area_house = house["area_m2"].sum()
        house_area_for_cluster.append(area_house)

    population_df = pd.DataFrame()
    population_df["cluster"] = clusters
    population_df.set_index("cluster", inplace=True)
    population_df["house_area_for_cluster"] = house_area_for_cluster
    people_for_cluster = (population_df["house_area_for_cluster"] * density).round()
    population_df["people_for_cluster"] = people_for_cluster

    people_for_cluster = population_df["people_for_cluster"]
    tier_pop_df = population_df["people_for_cluster"].apply(
        lambda x: pd.Series([x * y for y in tier_percent])
    )
    demand_tier_1 = pd.read_excel(input_file_profile_tier1)
    demand_tier_2 = pd.read_excel(input_file_profile_tier2)
    demand_tier_3 = pd.read_excel(input_file_profile_tier3)
    demand_tier_4 = pd.read_excel(input_file_profile_tier4)
    demand_tier_5 = pd.read_excel(input_file_profile_tier5)
    mean_demand_tier_df = pd.DataFrame()

    demand_tiers = [
        demand_tier_1,
        demand_tier_2,
        demand_tier_3,
        demand_tier_4,
        demand_tier_5,
    ]

    for i, demand_tier in enumerate(demand_tiers, start=1):
        mean_column_name = f"bus_{i}"
        mean_demand_tier_df[mean_column_name] = demand_tier["mean"]
    mean_demand_tier_df.insert(0, "bus_0", np.zeros(len(mean_demand_tier_df)))

    hours_index = pd.date_range(
        start="00:00:00", periods=len(mean_demand_tier_df), freq="H", normalize=True
    )
    mean_demand_tier_df.index = hours_index.time

    # Creazione di un DataFrame con tutti i tier e la std media oraria per ognuno
    std_demand_tier_df = pd.DataFrame()

    for i, demand_tier in enumerate(demand_tiers, start=1):
        mean_column_name = f"bus_{i}"
        std_demand_tier_df[mean_column_name] = demand_tier["std"]
    std_demand_tier_df.insert(0, "bus_0", np.zeros(len(mean_demand_tier_df)))

    std_demand_tier_df.index = hours_index.time

    date_range = pd.date_range(start="2013-01-01", end="2013-12-31", freq="D")
    yearly_mean_demand_tier_df = pd.concat(
        [mean_demand_tier_df] * len(date_range), ignore_index=True
    )
    date_time_index = pd.date_range(
        start="2013-01-01", end="2013-12-31 23:00:00", freq="H"
    )
    yearly_mean_demand_tier_df.index = date_time_index

    yearly_std_demand_tier_df = pd.concat(
        [std_demand_tier_df] * len(date_range), ignore_index=True
    )
    yearly_std_demand_tier_df.index = date_time_index

    dataframes_dict = {}
    for index, row in tier_pop_df.iterrows():
        loads = []
        for col in range(tier_pop_df.shape[1]):
            n = row[col]
            tier = yearly_mean_demand_tier_df.iloc[:, col]
            loads_h = n * tier
            loads.append(loads_h)

        loads_df = pd.DataFrame(loads)
        loads_df = loads_df.T

        nome_dataframe = f"bus_{index}"
        dataframes_dict[nome_dataframe] = loads_df

    dataframes_std_dict = {}
    for index, row in tier_pop_df.iterrows():
        loads = []
        for col in range(tier_pop_df.shape[1]):
            n = np.sqrt(row[col])
            tier = yearly_std_demand_tier_df.iloc[:, col]
            loads_h = n * tier
            loads.append(loads_h)

        loads_df = pd.DataFrame(loads)
        loads_df = loads_df.T
        loads_df = loads_df.applymap(lambda x: np.random.normal(0, x))

        nome_dataframe = f"bus_{index}"
        dataframes_std_dict[nome_dataframe] = loads_df

    loads_with_std = {}

    for i in range(len(dataframes_dict)):
        key_df = f"bus_{i}"
        key_std = f"bus_{i}"

        # Somma i dataframe corrispondenti
        if key_df in dataframes_dict and key_std in dataframes_std_dict:
            df = dataframes_dict[key_df] + dataframes_std_dict[key_std]
            loads_with_std[key_df] = df
        else:
            print(f"Keys {key_df} o {key_std} not in dictionary.")

    tot_loads_dict = {}
    for key in loads_with_std:
        load = loads_with_std[key]
        load_tot = load.sum(axis=1)

        total_loads_df = pd.DataFrame(load_tot, columns=[f"{key}"])
        tot_loads_dict[key] = total_loads_df

    tot_loads_df = pd.DataFrame()
    for key, cluster_load in tot_loads_dict.items():
        tot_loads_df = pd.concat([tot_loads_df, cluster_load], axis=1)
    tot_loads_df.to_csv(output_path_csv)


def calculate_load_ramp_std2(
    input_file_buildings,
    n,
    p,
    raster_path,
    shapes_path,
    sample_profile,
    output_file,
    input_file_profile_tier1,
    input_file_profile_tier2,
    input_file_profile_tier3,
    input_file_profile_tier4,
    input_file_profile_tier5,
    output_path_csv,
    tier_percent,
):

    cleaned_buildings = gpd.read_file(input_file_buildings)
    house = cleaned_buildings[cleaned_buildings["tags_building"] == "house"]
    area_tot = house["area_m2"].sum()

    pop_microgrid, microgrid_load = estimate_microgrid_population(
        n, p, raster_path, shapes_path, sample_profile, output_file
    )
    density = pop_microgrid / area_tot

    grouped_buildings = cleaned_buildings.groupby("cluster_id")
    clusters = np.sort(cleaned_buildings["cluster_id"].unique())
    house_area_for_cluster = []
    for cluster in clusters:
        cluster_buildings = pd.DataFrame(grouped_buildings.get_group(cluster))
        house = cluster_buildings[cluster_buildings["tags_building"] == "house"]
        area_house = house["area_m2"].sum()
        house_area_for_cluster.append(area_house)

    population_df = pd.DataFrame()
    population_df["cluster"] = clusters
    population_df.set_index("cluster", inplace=True)
    population_df["house_area_for_cluster"] = house_area_for_cluster
    people_for_cluster = (population_df["house_area_for_cluster"] * density).round()
    population_df["people_for_cluster"] = people_for_cluster

    people_for_cluster = population_df["people_for_cluster"]
    tier_pop_df = population_df["people_for_cluster"].apply(
        lambda x: pd.Series([x * y for y in tier_percent])
    )

    demand_tier_1 = pd.read_excel(input_file_profile_tier1)
    demand_tier_2 = pd.read_excel(input_file_profile_tier2)
    demand_tier_3 = pd.read_excel(input_file_profile_tier3)
    demand_tier_4 = pd.read_excel(input_file_profile_tier4)
    demand_tier_5 = pd.read_excel(input_file_profile_tier5)
    mean_demand_tier_df = pd.DataFrame()

    demand_tiers = [
        demand_tier_1,
        demand_tier_2,
        demand_tier_3,
        demand_tier_4,
        demand_tier_5,
    ]

    for i, demand_tier in enumerate(demand_tiers, start=1):
        mean_column_name = f"bus_{i}"
        mean_demand_tier_df[mean_column_name] = demand_tier["mean"]
    mean_demand_tier_df.insert(0, "bus_0", np.zeros(len(mean_demand_tier_df)))

    hours_index = pd.date_range(
        start="00:00:00", periods=len(mean_demand_tier_df), freq="H", normalize=True
    )
    mean_demand_tier_df.index = hours_index.time

    # Creazione di un DataFrame con tutti i tier e la std media oraria per ognuno
    std_demand_tier_df = pd.DataFrame()

    for i, demand_tier in enumerate(demand_tiers, start=1):
        mean_column_name = f"bus_{i}"
        std_demand_tier_df[mean_column_name] = demand_tier["std"]
    std_demand_tier_df.insert(0, "bus_0", np.zeros(len(mean_demand_tier_df)))

    std_demand_tier_df.index = hours_index.time

    date_range = pd.date_range(start="2013-01-01", end="2013-12-31", freq="D")
    yearly_mean_demand_tier_df = pd.concat(
        [mean_demand_tier_df] * len(date_range), ignore_index=True
    )
    date_time_index = pd.date_range(
        start="2013-01-01", end="2013-12-31 23:00:00", freq="H"
    )
    yearly_mean_demand_tier_df.index = date_time_index

    result_dict = {}
    for k in range(len(tier_pop_df)):
        pop_cluster = tier_pop_df.iloc[k, :]
        for j in range(len(pop_cluster)):
            num_iterations = int(pop_cluster[j])
            num_rows = len(mean_demand_tier_df)
            num_columns = num_iterations
            loads = np.zeros((num_rows, num_columns))
            for i in range(num_iterations):
                mean_load_person = mean_demand_tier_df.iloc[:, j].values
                std_load_person = np.random.normal(
                    0, mean_demand_tier_df.iloc[:, j].values[:num_rows]
                )
                load_person = mean_load_person + std_load_person
                loads[:, i] = load_person
            results_df = pd.DataFrame(loads)

            nome_dataframe = f"person_load_cluster_{k}_tier_{j}"
            result_dict[nome_dataframe] = results_df

    total_loads_cluster_tier_dict = {}

    for k in range(len(tier_pop_df)):
        key_cluster = f"{k}"
        total_loads_cluster_tier_dict[key_cluster] = {}

        for j in range(len(pop_cluster)):
            key_tier = f"{j}"
            load = result_dict[f"person_load_cluster_{key_cluster}_tier_{key_tier}"]
            load_tot = load.sum(axis=1)
            total_loads_df = pd.DataFrame(
                load_tot, columns=[f"bus_{key_cluster}_tier_{key_tier}"]
            )

            total_loads_cluster_tier_dict[key_cluster][key_tier] = total_loads_df

    total_loads_cluster = {}
    for k in range(len(tier_pop_df)):
        key_cluster = str(k)
        df0 = total_loads_cluster_tier_dict[key_cluster]["0"].values
        df1 = total_loads_cluster_tier_dict[key_cluster]["1"].values
        df2 = total_loads_cluster_tier_dict[key_cluster]["2"].values
        df3 = total_loads_cluster_tier_dict[key_cluster]["3"].values
        df4 = total_loads_cluster_tier_dict[key_cluster]["4"].values
        df5 = total_loads_cluster_tier_dict[key_cluster]["5"].values

        total_load = pd.DataFrame(
            df0 + df1 + df2 + df3 + df4 + df5, columns=[f"bus_{k}"]
        )
        total_loads_cluster[f"bus_{k}"] = total_load

    tot_loads_df = pd.DataFrame()
    for key, cluster_load in total_loads_cluster.items():
        tot_loads_df = pd.concat([tot_loads_df, cluster_load], axis=1)

    date_range = pd.date_range(start="2013-01-01", end="2013-12-31", freq="D")
    yearly_mean_demand_tier_df = pd.concat(
        [tot_loads_df] * len(date_range), ignore_index=True
    )
    date_time_index = pd.date_range(
        start="2013-01-01", end="2013-12-31 23:00:00", freq="H"
    )
    yearly_mean_demand_tier_df.index = date_time_index

    yearly_mean_demand_tier_df.to_csv(output_path_csv)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers_dist import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("build_demand")
        sets_path_to_root("pypsa-distribution")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.create_network)
    sample_profile = snakemake.input["sample_profile"]
    tier_percent = snakemake.params.tier["tier_percent"]

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
        snakemake.output["electric_load_2"],
        snakemake.input["building_csv"],
    )

    calculate_load_ramp(
        snakemake.input["clusters_with_buildings"],
        n,
        snakemake.config["load"]["scaling_factor"],
        worldpop_path,
        snakemake.input["microgrid_shapes"],
        sample_profile,
        snakemake.output["electric_load"],
        snakemake.input["profile_tier1"],
        snakemake.input["profile_tier2"],
        snakemake.input["profile_tier3"],
        snakemake.input["profile_tier4"],
        snakemake.input["profile_tier5"],
        snakemake.output["electric_load_3"],
        tier_percent,
    )

    calculate_load_ramp_std(
        snakemake.input["clusters_with_buildings"],
        n,
        snakemake.config["load"]["scaling_factor"],
        worldpop_path,
        snakemake.input["microgrid_shapes"],
        sample_profile,
        snakemake.output["electric_load"],
        snakemake.input["profile_tier1"],
        snakemake.input["profile_tier2"],
        snakemake.input["profile_tier3"],
        snakemake.input["profile_tier4"],
        snakemake.input["profile_tier5"],
        snakemake.output["electric_load_1"],
        tier_percent,
    )

    calculate_load_ramp_std2(
        snakemake.input["clusters_with_buildings"],
        n,
        snakemake.config["load"]["scaling_factor"],
        worldpop_path,
        snakemake.input["microgrid_shapes"],
        sample_profile,
        snakemake.output["electric_load"],
        snakemake.input["profile_tier1"],
        snakemake.input["profile_tier2"],
        snakemake.input["profile_tier3"],
        snakemake.input["profile_tier4"],
        snakemake.input["profile_tier5"],
        snakemake.output["electric_load"],
        tier_percent,
    )
