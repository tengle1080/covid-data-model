from typing import List
from datetime import datetime
from api.can_api_definition import (
    CovidActNowCountiesAPI,
    CovidActNowSummary,
    CovidActNowAreaSummary,
    CovidActNowTimeseries,
    CANPredictionTimeseriesRow,
    CANActualsTimeseriesRow,
    _Projections,
    _Actuals,
    _ResourceUsageProjection,
)
from libs.enums import Intervention
from libs.datasets import results_schema as rc
from libs.datasets.common_fields import CommonFields
from libs.datasets import can_model_output_schema as can_schema


def _format_date(input_date):
    if not input_date:
        raise Exception("Can't format a date that doesn't exist")
    if isinstance(input_date, str):
        # note if this is already in iso format it will be grumpy. maybe use dateutil
        datetime_obj = datetime.strptime(input_date, "%m/%d/%Y %H:%M")
        return datetime_obj
    if isinstance(input_date, datetime):
        return input_date
    raise Exception("Invalid date type when converting to api")


def _generate_projections(data: dict):
    projections = {
        'totalHospitalBeds': {
            'peakDate': data[rc.PEAK_HOSPITALIZATIONS],
            'shortageStartDate': data[rc.HOSPITAL_SHORTFALL_DATE],
            'peakShortfall': data[rc.PEAK_HOSPITALIZATION_SHORTFALL]
        },
        'ICUBeds': None,
        'Rt': data[rc.RT],
        'RtCI90': data[rc.RT_CI90],
    }
    return _Projections(**projections)


def _generate_actuals(data, intervention_str):
    hospital_beds = {
        "capacity": data[CommonFields.MAX_BED_COUNT],
        "currentUsage": data[CommonFields.CURRENT_HOSPITALIZED],
        "typicalUsageRate": data.get(CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE),
    }
    icu_beds = {
        "capacity": data[CommonFields.ICU_BEDS],
        "currentUsage": data[CommonFields.CURRENT_HOSPITALIZED],
        "typicalUsageRate": data.get(CommonFields.ICU_TYPICAL_OCCUPANCY_RATE),
    }

    return _Actuals(
        population=data.get(CommonFields.POPULATION),
        intervention=intervention_str,
        cumulativeConfirmedCases=data[CommonFields.CASES],
        cumulativeDeaths=data[CommonFields.DEATHS],
        cumulativePositiveTests=data[CommonFields.POSITIVE_TESTS],
        cumulativeNegativeTests=data[CommonFields.NEGATIVE_TESTS],
        hospitalBeds=hospital_beds,
        ICUBeds=icu_beds,
    )


def _generate_actuals_timeseries(timeseries_rows, intervention):
    result_rows = []

    for row in timeseries_rows:
        actuals = _generate_actuals(row, intervention.name)
        timeseries_actual = CANActualsTimeseriesRow(**actuals.dict(), date=row[CommonFields.DATE])
        result_rows.append(timeseries_actual)

    return result_rows


def _generate_prediction_row(json_data_row: dict):
    return CANPredictionTimeseriesRow(
        date=datetime.strptime(json_data_row[can_schema.DATE], "%m/%d/%y"),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=json_data_row[can_schema.ICU_BED_CAPACITY],
        ventilatorsInUse=json_data_row[can_schema.CURRENT_VENTILATED],
        ventilatorCapacity=json_data_row[can_schema.VENTILATOR_CAPACITY],
        RtIndicator=json_data_row[can_schema.RT_INDICATOR],
        RtIndicatorCI90=json_data_row[can_schema.RT_INDICATOR_CI90],
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
        cumulativePositiveTests=None,
        cumulativeNegativeTests=None,
    )


def generate_summary(
    latest: dict,
    projection_data: dict,
) -> CovidActNowSummary:
    projections = _generate_projections(projection_data)
    intervention = projection_data[CommonFields.INTERVENTION]
    actuals = _generate_actuals(latest, intervention)

    return CovidActNowSummary(
        lat=latest[CommonFields.LATITUDE],
        long=latest[CommonFields.LONGITUDE],
        actuals=actuals,
        stateName=latest[CommonFields.STATE_FULL_NAME],
        countyName=latest[CommonFields.COUNTY],
        fips=latest[CommonFields.FIPS],
        lastUpdatedDate=_format_date(projection_data[rc.LAST_UPDATED]),
        projections=projections,
    )


def generate_timeseries(
    projection_data: dict,
    model_timeseries: List[dict],
    historical_timeseries: List[dict],
    historical_latest: dict
) -> CovidActNowTimeseries:

    summary = generate_summary(historical_latest, projection_data)
    state_intervention = historical_latest[CommonFields.INTERVENTION]
    projection_timeseries = [_generate_prediction_row(data) for data in model_timeseries]
    actuals_timeseries = _generate_actuals_timeseries(
        historical_timeseries, state_intervention
    )

    return CovidActNowTimeseries(
        **summary.dict(),
        timeseries=projection_timeseries,
        actuals_timeseries=actuals_timeseries
    )
