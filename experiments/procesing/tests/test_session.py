import pytest
import pandas as pd
import numpy as np
from procesing.steps.session import (
    TemporalFeatureStep,
    BehavioralFeatureStep,
    ProductFeatureStep,
    UserAgentFeatureStep,
    ExtractSessionFeaturesStep,
    JoinLabelsStep,
    ValidateDataStep,
)


# TemporalFeatureStep tests
def test_temporal_empty(pipeline_context):
    result = TemporalFeatureStep(pipeline_context).transform(pd.DataFrame())
    assert 'sessionId' in result.columns
    assert result.empty


def test_temporal_basic(pipeline_context, session_interactions):
    result = TemporalFeatureStep(pipeline_context).transform(session_interactions)
    assert 'session_duration_sec' in result.columns
    assert 'interaction_velocity' in result.columns
    assert 'max_velocity_5min' in result.columns
    assert result['total_interactions'].sum() == len(session_interactions)


def test_temporal_timeout(pipeline_context):
    df = pd.DataFrame({
        'sessionId': ['s1', 's1'],
        'ts': ['2025-01-01T10:00:00Z', '2025-01-01T11:00:00Z'],  # 1 hour gap
    })
    result = TemporalFeatureStep(pipeline_context, timeout_sec=900).transform(df)
    assert result.iloc[0]['session_duration_sec'] == 0  # gap exceeds timeout


# BehavioralFeatureStep tests
def test_behavioral_empty(pipeline_context):
    result = BehavioralFeatureStep(pipeline_context).transform(pd.DataFrame())
    assert 'sessionId' in result.columns


def test_behavioral_counts(pipeline_context, session_interactions):
    result = BehavioralFeatureStep(pipeline_context).transform(session_interactions)
    assert 'page_views' in result.columns
    assert 'item_views' in result.columns
    assert 'hover_events' in result.columns
    assert result['total_events'].sum() == len(session_interactions)


def test_behavioral_hover_prefix(pipeline_context):
    df = pd.DataFrame({
        'sessionId': ['s1', 's1'],
        'eventName': ['hover_over_custom', 'hover_over_button'],
        'page': ['/products', '/products'],
    })
    result = BehavioralFeatureStep(pipeline_context).transform(df)
    assert result.iloc[0]['hover_events'] == 2


# ProductFeatureStep tests
def test_product_empty(pipeline_context):
    result = ProductFeatureStep(pipeline_context).transform(pd.DataFrame())
    assert 'sessionId' in result.columns


def test_product_features(pipeline_context, session_interactions):
    result = ProductFeatureStep(pipeline_context).transform(session_interactions)
    assert 'unique_products_viewed' in result.columns
    assert 'price_range' in result.columns
    assert result['unique_products_viewed'].sum() > 0


# UserAgentFeatureStep tests
def test_ua_empty(pipeline_context):
    result = UserAgentFeatureStep(pipeline_context).transform(pd.DataFrame())
    assert 'sessionId' in result.columns


def test_ua_headless_detection(pipeline_context):
    df = pd.DataFrame({
        'sessionId': ['s1', 's2'],
        'userAgent': ['Mozilla/5.0 Chrome/120', 'HeadlessChrome/120'],
    })
    result = UserAgentFeatureStep(pipeline_context).transform(df)
    assert 'is_headless' in result.columns
    headless = dict(zip(result['sessionId'], result['is_headless']))
    assert headless['s1'] == False
    assert headless['s2'] == True


def test_ua_browser_family(pipeline_context):
    df = pd.DataFrame({
        'sessionId': ['s1', 's2', 's3'],
        'userAgent': ['Mozilla/5.0 Firefox/120', 'Safari/605.1.15', 'Unknown'],
    })
    result = UserAgentFeatureStep(pipeline_context).transform(df)
    browsers = dict(zip(result['sessionId'], result['browser_family']))
    assert browsers['s1'] == 'Firefox'
    assert browsers['s2'] == 'Safari'
    assert browsers['s3'] == 'Other'


def test_ua_automation_detection(pipeline_context):
    df = pd.DataFrame({
        'sessionId': ['s1', 's2'],
        'userAgent': ['Selenium WebDriver', 'Normal Chrome/120'],
    })
    result = UserAgentFeatureStep(pipeline_context).transform(df)
    auto = dict(zip(result['sessionId'], result['is_automation']))
    assert auto['s1'] == True
    assert auto['s2'] == False


# ExtractSessionFeaturesStep tests
def test_extract_empty(pipeline_context):
    result = ExtractSessionFeaturesStep(pipeline_context).transform(pd.DataFrame())
    assert result.empty


def test_extract_merges_all(pipeline_context, session_interactions):
    result = ExtractSessionFeaturesStep(pipeline_context).transform(session_interactions)
    expected = ['session_duration_sec', 'total_events', 'unique_products_viewed', 'is_headless']
    for col in expected:
        assert col in result.columns
    assert 'experimentId' in result.columns


# JoinLabelsStep tests
def test_join_labels_tuple_input(pipeline_context):
    features = pd.DataFrame({'sessionId': ['s1'], 'experimentId': ['exp1'], 'total_events': [5]})
    experiments = pd.DataFrame({'id': ['exp1'], 'xp_human_only': [True]})
    result = JoinLabelsStep(pipeline_context).transform((features, experiments))
    assert 'is_agent' in result.columns
    assert result.iloc[0]['is_agent'] == False


def test_join_labels_empty_experiments(pipeline_context):
    features = pd.DataFrame({'sessionId': ['s1'], 'experimentId': ['exp1']})
    result = JoinLabelsStep(pipeline_context).transform((features, pd.DataFrame()))
    assert pd.isna(result.iloc[0]['is_agent'])


# ValidateDataStep tests
def test_validate_empty(pipeline_context):
    ValidateDataStep(pipeline_context).transform(pd.DataFrame())
    report = pipeline_context.get_cached('validation_report')
    assert report['status'] == 'empty'


def test_validate_missing_cols(pipeline_context):
    df = pd.DataFrame({'sessionId': ['s1'], 'ts': ['2025-01-01']})
    ValidateDataStep(pipeline_context).transform(df)
    report = pipeline_context.get_cached('validation_report')
    assert report['status'] == 'invalid'
    assert 'eventName' in report['missing_cols']


def test_validate_valid(pipeline_context, session_interactions):
    ValidateDataStep(pipeline_context).transform(session_interactions)
    report = pipeline_context.get_cached('validation_report')
    assert report['status'] == 'valid'
    assert report['sessions'] > 0
