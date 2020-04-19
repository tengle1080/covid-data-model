import logging
import math
import json
import datetime
import numbers

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from libs.CovidDatasets import JHUDataset as LegacyJHUDataset
from libs.datasets import JHUDataset
from libs.datasets import FIPSPopulation
from libs.datasets import DHBeds
from libs.datasets.dataset_utils import AggregationLevel

# from .epi_models.HarvardEpi import (
from .epi_models.TalusSEIR import (
    seir,
    dataframe_ify,
    generate_epi_params,
    generate_r0,
    brute_force_r0,
    L,
)

_logger = logging.getLogger(__name__)

### TODO: need to tie together all the interventions and actuals and get a final
# results set
class ModelRun:
    def __init__(self, state, country="USA", county=None):
        self.state = state
        self.country = country
        self.county = county

        # define constants used in model parameter calculations
        self.observed_daily_growth_rate = 1.17
        self.days_to_model = 365

        # when going back to test hypothetical intervnetions in the past,
        # use this to start the data from this date instead of latest reported
        self.override_model_start = False

        ## Variables for calculating model parameters Hill -> our names/calcs
        # IncubPeriod: Average incubation period, days - presymptomatic_period
        # DurMildInf: Average duration of mild infections, days - duration_mild_infections
        # FracMild: Average fraction of (symptomatic) infections that are mild - (1 - hospitalization_rate)
        # FracSevere: Average fraction of (symptomatic) infections that are severe - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # FracCritical: Average fraction of (symptomatic) infections that are critical - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # CFR: Case fatality rate (fraction of infections that eventually result in death) - case_fatality_rate
        # DurHosp: Average duration of hospitalization (time to recovery) for individuals with severe infection, days - hospital_time_recovery
        # TimeICUDeath: Average duration of ICU admission (until death or recovery), days - icu_time_death

        # LOGIC ON INITIAL CONDITIONS:
        # hospitalized = case load from timeseries on last day of data / 4
        # mild = hospitalized / hospitalization_rate
        # icu = hospitalized * hospitalized_cases_requiring_icu_care
        # expoosed = exposed_infected_ratio * mild

        # Time before exposed are infectious (days)
        self.presymptomatic_period = 3

        # Time mildly infected people stay sick before
        # hospitalization or recovery (days)
        self.duration_mild_infections = 6

        # Time asymptomatically infected people stay
        # infected before recovery (days)
        self.duration_asymp_infections = 6

        # Duration of hospitalization before icu or
        # recovery (days)
        self.hospital_time_recovery = 6

        # Time from ICU admission to death (days)
        self.icu_time_death = 8

        ####################################################
        # BETA: transmission rate (new cases per day).
        # The rate at which infectious cases of various
        # classes cause secondary or new cases.
        ####################################################
        #
        # Transmission rate of infected people with no
        # symptoms [A] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.6
        # Current: Calculated based on observed doubling
        # rates
        self.beta_asymp = 0.3 + ((self.observed_daily_growth_rate - 1.09) / 0.02) * 0.05
        #
        # Transmission rate of infected people with mild
        # symptoms [I_1] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.6
        # Current: Calculated based on observed doubling
        # rates
        self.beta = 0.3 + ((self.observed_daily_growth_rate - 1.09) / 0.02) * 0.05
        #
        # Transmission rate of infected people with severe
        # symptoms [I_2] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.1
        self.beta_hospitalized = 0.1
        #
        # Transmission rate of infected people with severe
        # symptoms [I_3] (new cases per day)
        # This is really beta * N, but it's easier to talk about this way
        # Default: 0.1
        self.beta_icu = 0.1
        #
        ####################################################

        # Pecentage of asymptomatic, infectious [A] people
        # out of all those who are infected
        # make 0 to remove this stock
        self.percent_asymp = 0.3

        self.percent_infectious_symptomatic = 1 - self.percent_asymp

        self.hospitalization_rate = 0.10
        self.hospitalized_cases_requiring_icu_care = 0.25

        self.percent_symptomatic_mild = (
            self.percent_infectious_symptomatic - self.hospitalization_rate
        )

        # changed this from CFR to make the calc of mu clearer
        self.death_rate_for_critical = 0.4

        # CFR is calculated from the input parameters vs. fixed
        self.case_fatality_rate = (
            (1 - self.percent_asymp)
            * self.hospitalization_rate
            * self.hospitalized_cases_requiring_icu_care
            * self.death_rate_for_critical
        )

        # if true we calculatied the exposed initial stock from the infected number vs. leaving it at 0
        self.exposed_from_infected = True
        self.exposed_infected_ratio = 1

        # different ways to model the actual data

        # cases represent all infected symptomatic
        # based on proportion of mild/hospitalized/icu
        # described in params
        # self.model_cases = "divided_into_infected"

        # 1/4 cases are hopsitalized, mild and icu
        # based on proporition of hopsitalized
        # described in params
        self.model_cases = "one_in_4_hospitalized"

        self.hospital_capacity_change_daily_rate = 1.05
        self.max_hospital_capacity_factor = 2.07
        self.initial_hospital_bed_utilization = 0.6
        self.case_fatality_rate_hospitals_overwhelmed = (
            self.hospitalization_rate * self.hospitalized_cases_requiring_icu_care
        )

        self.interventions = {}

    class SnapShot:
        def __init__(self, model_run, type):
            self.N = model_run.population

            # this is an inital run or a past run, will have to build the initial
            # conditions from the timeseries data
            if type == "base":
                if model_run.model_cases == "divided_into_infected":
                    cases = model_run.past_data.get(key="cases", default=0)
                    self.hospitalized = cases * model_run.hospitalization_rate
                    self.icu = (
                        self.hospitalized
                        * model_run.hospitalized_cases_requiring_icu_care
                    )
                    self.mild = cases - self.hospitalized - self.icu
                    self.asymp = self.mild * model_run.percent_asymp
                    self.dead = model_run.past_data.get(key="deaths", default=0)
                elif model_run.model_cases == "one_in_4_hospitalized":
                    self.hospitalized = (
                        model_run.past_data.get(key="cases", default=0) / 4
                    )
                    self.mild = self.hospitalized / model_run.hospitalization_rate
                    self.icu = (
                        self.hospitalized
                        * model_run.hospitalized_cases_requiring_icu_care
                    )
                    self.asymp = self.mild * model_run.percent_asymp
                    self.dead = model_run.past_data.get(key="deaths", default=0)
            elif type in ("intervention", "past-actual", "past-counterfactual"):
                # this should be an intervention run, so the initial conditions are more
                # fleshed out
                self.mild = model_run.past_data.get(key="infected_a", default=0)
                self.hospitalized = model_run.past_data.get(key="infected_b", default=0)
                self.icu = model_run.past_data.get(key="infected_c", default=0)
                self.asymp = model_run.past_data.get(key="asymp", default=0)
                self.dead = model_run.past_data.get(key="dead", default=0)

            self.exposed = model_run.exposed_infected_ratio * self.mild
            self.infected = self.asymp + self.mild + self.hospitalized + self.icu
            self.recovered = model_run.past_data.get(key="recovered", default=0)
            susceptible = self.N - (self.infected + self.recovered + self.dead)

            self.y0 = [
                int(self.exposed),
                int(self.mild),
                int(self.hospitalized),
                int(self.icu),
                int(self.recovered),
                int(self.dead),
                int(self.asymp),
            ]

    # use only if you're doing a stand-alone run, if you're doing a lot of regions
    # then grab all the data and just call get_data_subset for each run
    def get_data(self, min_date):
        # TODO rope in counties

        self.min_date = min_date

        beds = DHBeds.local().beds()
        population_data = FIPSPopulation.local().population()

        timeseries = (
            JHUDataset.local()
            .timeseries()
            .get_subset(
                AggregationLevel.STATE,
                after=min_date,
                country=self.country,
                state=self.state,
            )
        )

        if self.county is None:
            self.population = population_data.get_state_level(self.country, self.state)
            self.beds = beds_data.get_beds_by_country_state(self.country, self.state)
            self.timeseries = timeseries.get_data(state=self.state)
        else:
            # do county thing
            pass

        if self.override_model_start is False:
            self.start_date = self.timeseries.loc[
                (self.timeseries["cases"] > 0), "date"
            ].max()
        else:
            self.start_date = self.override_model_start

        return

    def process_actuals(self, actuals):

        if self.model_run.model_cases == "divided_into_infected":
            actuals.loc[:, "infected_b"] = (
                actuals["cases"] * self.model_run.hospitalization_rate
            )
            actuals.loc[:, "infected_c"] = (
                actuals["infected_b"]
                * self.model_run.hospitalized_cases_requiring_icu_care
            )
            actuals.loc[:, "infected_a"] = (
                actuals["cases"] - actuals["infected_b"] - actuals["infected_c"]
            )
            actuals.loc[:, "asymp"] = (
                actuals["infected_a"] * self.model_run.percent_asymp
            )
            actuals.loc[:, "dead"] = actuals["deaths"]
        elif self.model_run.model_cases == "one_in_4_hospitalized":
            actuals.loc[:, "infected_b"] = actuals["cases"] / 4
            actuals.loc[:, "infected_a"] = (
                actuals["infected_b"] / self.model_run.hospitalization_rate
            )
            actuals.loc[:, "infected_c"] = (
                actuals["infected_b"]
                * self.model_run.hospitalized_cases_requiring_icu_care
            )
            actuals.loc[:, "asymp"] = (
                actuals["infected_a"] * self.model_run.percent_asymp
            )
            actuals.loc[:, "dead"] = actuals["deaths"]

        actuals.loc[:, "exposed"] = (
            self.model_run.exposed_infected_ratio * actuals["infected_a"]
        )

        match_columns = [
            "date",
            "exposed",
            "infected_a",
            "infected_b",
            "infected_c",
            "recovered",
            "dead",
            "asymp",
        ]
        return actuals.loc[:, match_columns]

    def get_data_subset(
        self, beds_data, population_data, timeseries, min_date,
    ):
        # TODO rope in counties

        self.min_date = min_date

        timeseries = timeseries.get_subset(
            AggregationLevel.STATE,
            after=self.min_date,
            country=self.country,
            state=self.state,
        )

        if self.county is None:
            self.population = population_data.get_state_level(self.country, self.state)
            self.beds = beds_data.get_state_level(self.state)
            self.timeseries = timeseries.get_data(state=self.state)
        else:
            # do county thing
            pass

        if self.override_model_start is False:
            self.start_date = self.timeseries.loc[
                (self.timeseries["cases"] > 0), "date"
            ].max()
        else:
            self.start_date = self.override_model_start

        self.actuals = self.timeseries.loc[(self.timeseries.date < self.start_date), :]
        self.raw_actuals = self.actuals.copy()

        # get a series of the relevant row in the df
        self.past_data = self.timeseries.loc[
            (self.timeseries.date == self.start_date), :
        ].iloc[0]

        self.default_past_data = self.past_data.copy()

        return

    def set_epi_model(self, epi_model_type):
        if epi_model_type == "seir":
            self.model_type = "seir"

            self.model_cols = [
                "total",
                "susceptible",
                "exposed",
                "infected",
                "infected_a",
                "infected_b",
                "infected_c",
                "recovered",
                "dead",
            ]

            from libs.epi_models import HarvardEpi as EpiModel

        elif epi_model_type == "asymp":
            self.model_type = "asymp"

            from libs.epi_models.TalusSEIRClass import TalusSEIR as EpiRun

        self.epi_run = EpiRun("base", self)
        self.epi_run.generate_epi_params()
        self.epi_run.InitConditions = self.SnapShot(self, "base")

    def reload_params(self):
        self.results_list = []
        self.display_df = None

        self.actuals = self.raw_actuals

        self.epi_run.EpiParameters = None
        self.epi_run.InitConditions = None

        self.past_data = self.default_past_data

        self.epi_run.InitConditions = self.SnapShot(self, "base")

        self.epi_run.generate_epi_params()

    def run(self):
        self.epi_run.seir()
        self.epi_run.dataframe_ify()

        self.results_list = [self.epi_run.display_df.copy()]

    def add_intervention(self, intervention):
        if self.model_type == "seir":
            from libs.epi_models.HarvardEpi import Intervention

        elif self.model_type == "asymp":
            from libs.epi_models.TalusSEIRClass import Intervention

        intervention_name = f"intervention_{self.state}_{intervention['name']}"

        intervention["system_name"] = intervention_name

        int_object = Intervention(intervention, self)
        int_object.InitConditions = self.SnapShot(self, intervention["type"])

        self.interventions[intervention_name] = int_object

    def run_all_interventions(self):

        sorted_interventions = sorted(
            list(self.interventions.values()),
            key=lambda intervention: intervention.intervention[
                "intervention_start_date"
            ],
        )

        for intervention in sorted_interventions:
            self.run_intervention(intervention.intervention["system_name"])

    def get_prior_run(self, name):
        prior_run = self.results_list[-1].copy()

        print(
            prior_run.loc[
                prior_run["date"] == self.interventions[name].intervention_start_date
            ]
        )

        return prior_run

    def run_intervention(self, name):
        # set the initial conditions based on the prior run
        self.interventions[name].set_prior_run(self.get_prior_run(name))

        self.past_data = self.interventions[name].initial_conditions

        self.interventions[name].load_epi()
        self.interventions[name].InitConditions = self.SnapShot(
            self, self.interventions[name].type
        )

        self.interventions[name].seir()
        self.interventions[name].dataframe_ify()

        # print('prior run value counts:')
        # print(self.interventions[name].display_df.source.value_counts())

        self.results_list.append(self.interventions[name].display_df.copy())

        # print('value_counts for all results')
        # for i, results in enumerate(self.results_list):
        #    print(f'item {i} in list')
        #    print(results.source.value_counts())

    def drop_interventions(self):
        self.interventions = {}
        self.run()


def plot_df(df_to_plot, cols, title="", y_max=8000000):
    cols.append("date")

    df_to_plot = df_to_plot.loc[:, cols]

    line_day = datetime.datetime.now() - datetime.timedelta(days=2)

    x_dates = df_to_plot["date"].dt.strftime("%Y-%m-%d").sort_values().unique()

    df_to_plot.set_index("date", inplace=True)

    stacked = df_to_plot.stack().reset_index()

    stacked.columns = ["date", "Population", "Number of people"]

    plt.figure(figsize=(15, 8))

    plt.axvline(line_day, 0, y_max, linestyle="--")
    plt.ylim(0, y_max)

    plt.title(title)

    df_plt = sb.lineplot(x="date", y="Number of people", hue="Population", data=stacked)
    # df_plt.set_xticklabels(labels=x_dates, rotation=45, ha='right')

    return df_plt


def prep_plot(prep_df, chart_cols, title, y_max=8000000):
    prep_df["date"] = pd.to_datetime(prep_df["date"])

    first_case_date = prep_df.loc[(prep_df.infected > 0), "date"].min()
    peak_date = prep_df.loc[(prep_df.infected_b == prep_df.infected_b.max())][
        "date"
    ].values[0]
    peak = prep_df.loc[(prep_df.infected_b == prep_df.infected_b.max())][
        "infected_b"
    ].values[0]

    icu_peak_date = prep_df.loc[(prep_df.infected_c == prep_df.infected_c.max())][
        "date"
    ].values[0]
    icu_peak = prep_df.loc[(prep_df.infected_c == prep_df.infected_c.max())][
        "infected_c"
    ].values[0]

    deaths = prep_df.loc[:, "dead"].max()

    print("first case")
    print(first_case_date)
    print("peak in hospitalizations")
    print(peak_date)
    print(f"{peak:,}")
    print("peak in icu")
    print(icu_peak_date)
    print(f"{icu_peak:,}")
    print("deaths")
    print(f"{deaths:,}")

    plot_df(
        prep_df,
        chart_cols,
        f"{title}. Peak hospitalizations: {int(peak):,}. Deaths: {int(deaths):,}",
        y_max,
    )


def plot_actuals(model_df, actuals_df, model_cols, title, y_max=8000000):

    combo_df = pd.merge(model_df, actuals_df, how="outer", on="date")

    plot_df(
        combo_df, model_cols, f"{title}.", y_max,
    )


def report_months(df):
    date_list = [
        datetime.datetime(2020, 5, 1).date(),
        datetime.datetime(2020, 6, 1).date(),
        datetime.datetime(2020, 7, 1).date(),
        datetime.datetime(2020, 8, 1).date(),
        datetime.datetime(2020, 9, 1).date(),
        datetime.datetime(2020, 10, 1).date(),
        datetime.datetime(2020, 11, 1).date(),
        datetime.datetime(2020, 12, 1).date(),
    ]

    cols = {
        "date": "Date",
        "infected_a": "Infected",
        "infected_b": "Hospitalized",
        "infected_c": "ICU",
        "dead": "Deaths",
    }

    report_df = df.loc[(df["date"].isin(date_list)), list(cols.keys())]

    report_df.rename(columns=cols, inplace=True)

    return report_df.T
