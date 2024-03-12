def ensure_time_progression(last_event_time, new_event_time):
    """
    Ensures that the new event time is strictly greater than the last event time.
    
    Parameters:
    - last_event_time: The time of the last event in the sequence.
    - new_event_time: The initially generated time for the new event.
    
    Returns:
    - Adjusted time for the new event, ensuring time progression.
    """
    if (last_event_time - new_event_time) > 0:
        # If the new event time is not strictly greater, adjust it
        new_event_time = last_event_time + (last_event_time - new_event_time)

    return new_event_time
