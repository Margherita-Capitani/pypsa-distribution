# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from _helpers_dist import configure_logging, create_logger, read_osm_config
from ramp import Appliance, UseCase, User, get_day_type


def create_demand_profile(
    days,
    start,
    xlsx_input_path,
    excel_profiles_output_path,
    excel_daily_profile_output_path,
):
    use_case = UseCase()
    use_case.load(xlsx_input_path)

    n_days = days
    num_users_values = []
    for user in use_case.users:
        num_users_values.append(user.num_users)
    n_users = num_users_values
    date_start = start
    use_case.date_start = date_start
    use_case.initialize(num_days=n_days, force=True)
    data = use_case.generate_daily_load_profiles(flat=True)
    data = data / n_users

    profile = pd.DataFrame(
        data=data,
        index=pd.date_range(start=date_start, periods=1440 * n_days, freq="T"),
    )

    # Reshape to obtain hourly average values:
    data = profile.iloc[:, 0].values
    group_size = 60
    n_groups = len(data) // group_size
    hourly_profile = data.reshape(-1, group_size).mean(axis=1)

    # Reshape to get a data frame divided into days:
    group_size = 24
    num_groups = len(hourly_profile) // group_size
    daily_profile = hourly_profile.reshape(num_groups, group_size).T
    daily_profile = pd.DataFrame(
        daily_profile, columns=[f"Day_{i+1}" for i in range(num_groups)]
    )
    date_index = pd.date_range(start="00:00", periods=24, freq="1H").time
    daily_profile.index = date_index

    # Calculation of the mean value and standard deviation to represent a typical day
    daily_h_mean = daily_profile.mean(axis=1)
    daily_h_std = daily_profile.std(axis=1)

    daily_type = pd.DataFrame({"mean": daily_h_mean, "std": daily_h_std})
    daily_type.index = date_index

    daily_profile.to_excel(excel_profiles_output_path)
    daily_type.to_excel(excel_daily_profile_output_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers_dist import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("ramp_build_demand_profile", user_type="School")
        sets_path_to_root("pypsa-distribution")
    configure_logging(snakemake)

    days = snakemake.params.ramp["days"]
    date_start = snakemake.params.snapshoots["start"]

    create_demand_profile(
        days,
        date_start,
        snakemake.input["user_description"],
        snakemake.output["daily_demand_profiles"],
        snakemake.output["daily_type_demand_profile"],
    )
