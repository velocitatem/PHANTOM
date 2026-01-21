"""Unified state representation utilities for MDP state encoding
Used by both experiments/ and sim/ components for consistent state handling
"""
from typing import Any, Callable


def make_state_repr(page: str = None, product_id: str = None, event_name: str = None) -> str:
    """create canonical state representation string from components
    format: page|productId|eventName
    """
    p = page or 'unk'
    pid = product_id or 'none'
    en = event_name or 'unknown'
    return f"{p}|{pid}|{en}"


def event_to_state(evt: Any) -> str:
    """convert event object/dict to state string
    supports both object attributes and dict keys
    """
    if isinstance(evt, dict):
        return make_state_repr(
            page=evt.get('page'),
            product_id=evt.get('productId'),
            event_name=evt.get('eventName') or evt.get('event_type')
        )
    return make_state_repr(
        page=getattr(evt, 'page', None),
        product_id=getattr(evt, 'productId', None),
        event_name=getattr(evt, 'eventName', None) or getattr(evt, 'event_type', None)
    )


def parse_state(state_str: str) -> dict:
    """parse state string back to components
    returns: {'page': str, 'productId': str, 'eventName': str}
    """
    parts = state_str.split('|')
    return {
        'page': parts[0] if len(parts) > 0 and parts[0] != 'unk' else None,
        'productId': parts[1] if len(parts) > 1 and parts[1] != 'none' else None,
        'eventName': parts[2] if len(parts) > 2 and parts[2] != 'unknown' else None
    }


def get_event_name(evt: Any) -> str:
    """extract event name from event object/dict"""
    if isinstance(evt, dict):
        return evt.get('eventName') or evt.get('event_type') or ''
    return getattr(evt, 'eventName', None) or getattr(evt, 'event_type', None) or ''


def get_timestamp(evt: Any) -> Any:
    """extract timestamp from event object/dict"""
    if isinstance(evt, dict):
        return evt.get('ts') or evt.get('timestamp')
    return getattr(evt, 'ts', None) or getattr(evt, 'timestamp', None)


def create_state_fn() -> Callable:
    """factory for state representation function"""
    return event_to_state


def create_event_name_fn() -> Callable:
    """factory for event name extraction function"""
    return get_event_name


def create_timestamp_fn() -> Callable:
    """factory for timestamp extraction function (returns raw value, use features.parse_timestamp to convert)"""
    return get_timestamp
