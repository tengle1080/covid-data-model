
class CommonFields(object):
    """Common field names shared across different sources of data"""

    FIPS = "fips"

    # 2 letter state abbreviation, i.e. MA
    STATE = "state"

    COUNTRY = "country"

    COUNTY = "county"

    AGGREGATE_LEVEL = "aggregate_level"

    DATE = "date"

    # Full state name, i.e. Massachusetts
    STATE_FULL_NAME = "state_full_name"

    CASES = "cases"
    DEATHS = "deaths"
    RECOVERED = "recovered"
    CUMULATIVE_HOSPITALIZED = "cumulative_hospitalized"
    CUMULATIVE_ICU = "cumulative_icu"

    POSITIVE_TESTS = "positive_tests"
    NEGATIVE_TESTS = "negative_tests"

    # Current values
    CURRENT_ICU = "current_icu"
    CURRENT_HOSPITALIZED = "current_hospitalized"
    CURRENT_VENTILATED = "current_ventilated"

    POPULATION = "population"

    STAFFED_BEDS = "staffed_beds"
    LICENSED_BEDS = "licensed_beds"
    ICU_BEDS = "icu_beds"
    ALL_BED_TYPICAL_OCCUPANCY_RATE = "all_beds_occupancy_rate"
    ICU_TYPICAL_OCCUPANCY_RATE = "icu_occupancy_rate"
    MAX_BED_COUNT = "max_bed_count"

    INTERVENTION = "intervention"


class CommonIndexFields(object):
    # Column for FIPS code. Right now a column containing fips data may be
    # county fips (a length 5 string) or state fips (a length 2 string).
    FIPS = CommonFields.FIPS

    # 2 letter state abbreviation, i.e. MA
    STATE = CommonFields.STATE

    COUNTRY = CommonFields.COUNTRY

    AGGREGATE_LEVEL = CommonFields.AGGREGATE_LEVEL

    DATE = CommonFields.DATE


class ModelFields(object):
    TOTAL = "total"          # All people in the model. This should always be population.
    TOTAL_SUSCEPTIBLE = "susceptible"
    EXPOSED = "exposed"
    INFECTED = "infected"
    # Infected by not hospitalized
    INFECTED_A = "infected_a"

    # Hospitalized but not ICU
    INFECTED_B = "infected_b"
    # In ICU
    INFECTED_C = "infected_c"

    # Total hospitalized
    ALL_HOSPITALIZED = "all_hospitalized"

    # Total infected (in hospital or not)
    ALL_INFECTED = "all_infected"

    DEAD = "dead"

    # General bed capacity excluding ICU beds.
    BEDS = "beds"

    CUMULATIVE_INFECTED = "cumulative_infected"

    # Effective reproduction number at time t.
    Rt = 'Rt'

    # 90% confidence interval at time t.
    Rt_ci90 = 'Rt_ci90'

    CURRENT_VENTILATED = 'current_ventilated'

    POPULATION = "population"

    ICU_BED_CAPACITY = "icu_bed_capacity"

    VENTILATOR_CAPACITY = "ventilator_capacity"

    RT_INDICATOR = 'Rt_indicator'

    RT_INDICATOR_CI90 = 'Rt_indicator_ci90'

    FIPS = "fips"
    STATE = "state"
    AGGREGATE_LEVEL = "aggregate_level"
    COUNTRY = "country"
    INTERVENTION = "intervention"
