from hypothesis import given, strategies as st
import time
import pytest
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError

@given(st.lists(st.booleans(), max_size=100))
def test_circuit_breaker_never_skips_half_open(outcomes: list[bool]) -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=3, reset_timeout_seconds=0.1, success_threshold=2)
    
    for success in outcomes:
        # Check invariants
        if breaker.state == CircuitState.OPEN:
            # Should deny request unless timeout passed
            # Let's mock time passing randomly
            pass
            
        # Try call
        def dummy_call():
            if not success:
                raise ValueError("Simulated failure")
            return "ok"
            
        try:
            prev_state = breaker.state
            breaker.call(dummy_call)
        except (ValueError, CircuitOpenError):
            pass
            
        # Invariant 1: If it was OPEN, it can only transition to HALF_OPEN (after timeout)
        # However, call() automatically throws CircuitOpenError if OPEN, so state shouldn't jump to CLOSED
        assert not (prev_state == CircuitState.OPEN and breaker.state == CircuitState.CLOSED)

@given(st.integers(min_value=1, max_value=10))
def test_circuit_breaker_opens_after_n_failures(n: int) -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=n, reset_timeout_seconds=1.0)
    
    def fail_call():
        raise ValueError("fail")
        
    for _ in range(n - 1):
        try:
            breaker.call(fail_call)
        except ValueError:
            pass
        assert breaker.state == CircuitState.CLOSED
        
    # Nth failure should open it
    try:
        breaker.call(fail_call)
    except ValueError:
        pass
        
    assert breaker.state == CircuitState.OPEN
