from __future__ import annotations

from dataclasses import dataclass

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitOpenError
from reliability_lab.providers import FakeLLMProvider, ProviderError, ProviderResponse


@dataclass(slots=True)
class GatewayResponse:
    text: str
    route: str
    provider: str | None
    cache_hit: bool
    latency_ms: float
    estimated_cost: float
    error: str | None = None


class ReliabilityGateway:
    """Routes requests through cache, circuit breakers, and fallback providers."""

    def __init__(
        self,
        providers: list[FakeLLMProvider],
        breakers: dict[str, CircuitBreaker],
        cache: ResponseCache | SharedRedisCache | None = None,
        budget_limit: float = 0.5, # Example budget limit
    ):
        self.providers = providers
        self.breakers = breakers
        self.cache = cache
        self.budget_limit = budget_limit
        self.cumulative_cost = 0.0

    def complete(self, prompt: str) -> GatewayResponse:
        """Return a reliable response or a static fallback."""
        if self.cache is not None:
            cached_text, score = self.cache.get(prompt)
            if cached_text is not None:
                return GatewayResponse(
                    text=cached_text,
                    route=f"cache_hit:{score:.2f}",
                    provider=None,
                    cache_hit=True,
                    latency_ms=0.0,
                    estimated_cost=0.0,
                )
                
        last_error = None
        
        # Cost-aware routing logic
        if self.cumulative_cost >= self.budget_limit:
            return GatewayResponse(
                text="The service has exceeded its cost budget. Only cached responses are available.",
                route="static_fallback",
                provider=None,
                cache_hit=False,
                latency_ms=0.0,
                estimated_cost=0.0,
                error="budget_exceeded",
            )
            
        for i, provider in enumerate(self.providers):
            # If we are over 80% budget, skip expensive providers (e.g. the first one)
            if self.cumulative_cost >= 0.8 * self.budget_limit and i == 0:
                continue
                
            breaker = self.breakers[provider.name]
            try:
                response = breaker.call(provider.complete, prompt)
                if self.cache is not None:
                    self.cache.set(prompt, response.text, {"provider": provider.name})
                
                self.cumulative_cost += response.estimated_cost
                route = "primary" if i == 0 else "fallback"
                return GatewayResponse(
                    text=response.text,
                    route=route,
                    provider=provider.name,
                    cache_hit=False,
                    latency_ms=response.latency_ms,
                    estimated_cost=response.estimated_cost,
                )
            except (ProviderError, CircuitOpenError) as e:
                last_error = str(e)
                continue
                
        return GatewayResponse(
            text="The service is temporarily degraded. Please try again soon.",
            route="static_fallback",
            provider=None,
            cache_hit=False,
            latency_ms=0.0,
            estimated_cost=0.0,
            error=last_error,
        )
