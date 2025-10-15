import requests, time, hashlib
import os
from datetime import date, datetime
from dotenv import load_dotenv
from typing import Dict, Optional, Union, Any, List, Tuple, Callable, Iterable, Mapping
import json
from functools import lru_cache

# generate authorisation header - this is verified against expedia's EPS signature generator at https://developers.expediagroup.com/rapid/resources/tools/signature-generator
def auth_header(API_KEY, SHARED_SECRET):
    """Return the EPS signature header for the provided key/secret pair."""
    ts = str(int(time.time()))
    sig = hashlib.sha512((API_KEY + SHARED_SECRET + ts).encode("utf-8")).hexdigest()
    return f"EAN APIKey={API_KEY},Signature={sig},timestamp={ts}"

class ExpediaAPIError(RuntimeError):
    """Raised when the Expedia API responds with an error or unexpected payload."""

class ExpediaClient:
    """
    Minimal EPS client that encapsulates auth signing, default headers, and
    convenience helpers.
    """

    DEFAULT_TIMEOUT = 30
    DEFAULT_USER_AGENT = "RapidTest/0.1"
    DEFAULT_CUSTOMER_IP = "8.8.8.8"  # any valid public IP

    RATE_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
        "pub_prepay": {"payment_terms": None, "partner_point_of_sale": "SA_MOD"},
        "pub_hotelpay": {"payment_terms": None, "partner_point_of_sale": "HC_MOD"},
        "mkt_prepay": {"payment_terms": "EPSMOR_MARKET", "partner_point_of_sale": "EPSMOR_SA_MARKET"},
        "mkt_hotelpay": {"payment_terms": "CCC_MARKET", "partner_point_of_sale": "CCC_SA_HC_MARKET"},
        "priv_pkg": {"payment_terms": "PKG_EPSMOR", "partner_point_of_sale": "EPSMOR_PKG_EXPPKG_MOD"},
    }

    def __init__(
        self,
        api_key: str,
        shared_secret: str,
        base_url: str,
        *,
        session: Optional[requests.Session] = None,
        user_agent: Optional[str] = None,
        customer_ip: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialise the client with credentials, base URL, and optional overrides."""
        if not api_key or not shared_secret or not base_url:
            raise ValueError("api_key, shared_secret, and base_url are all required.")

        self.api_key = api_key
        self.shared_secret = shared_secret
        self.base_url = base_url.rstrip("/") + "/"
        self.session = session or requests.Session()
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.customer_ip = customer_ip or self.DEFAULT_CUSTOMER_IP
        self.timeout = timeout or self.DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls, *, load: bool = True, **kwargs: Any) -> "ExpediaClient":
        """Construct a client using credentials pulled from environment variables."""
        if load:
            load_dotenv()

        api_base = os.getenv("EXPEDIA_API_BASE")
        api_key = os.getenv("EXPEDIA_API_KEY")
        shared_secret = os.getenv("EXPEDIA_SHARED_SECRET")

        if not api_base or not api_key or not shared_secret:
            raise ExpediaAPIError("Missing Expedia credentials in environment variables.")

        return cls(api_key, shared_secret, api_base, **kwargs)

    # --- internal helpers -------------------------------------------------
    def _authorization_header(self) -> str:
        """Build the Authorization header for the current client credentials."""
        return auth_header(self.api_key, self.shared_secret)

    def _build_headers(
        self,
        extra: Optional[Dict[str, str]] = None,
        *,
        customer_ip: Optional[str] = None,
    ) -> Dict[str, str]:
        """Compose default headers plus any extra overrides for an API request."""
        headers = {
            "Authorization": self._authorization_header(),
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": self.user_agent,
            "Customer-Ip": customer_ip or self.customer_ip,
        }
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Union[Dict[str, Any], List[Tuple[str, Any]]]] = None,
        json_payload: Optional[Any] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        customer_ip: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Execute an HTTP request against the EPS API and return parsed JSON data."""
        url = self.base_url + path.lstrip("/")
        headers = self._build_headers(extra_headers, customer_ip=customer_ip)
        try:
            resp = self.session.request(
                method.upper(),
                url,
                params=params,
                json=json_payload,
                headers=headers,
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise ExpediaAPIError(f"Request to {url} failed: {exc}") from exc

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body_preview = ""
            try:
                body_preview = json.dumps(resp.json(), indent=2)
            except ValueError:
                body_preview = (resp.text or "").strip()
            raise ExpediaAPIError(
                f"Expedia API error ({resp.status_code}) for {method.upper()} {url}: {body_preview}"
            ) from exc

        if not resp.content:
            return {}

        try:
            return resp.json()
        except ValueError as exc:
            raise ExpediaAPIError(f"Non-JSON response for {method.upper()} {url}") from exc

    @staticmethod
    def _to_unique_dict(
        rows: List[Dict[str, Any]],
        *,
        key: str = "property_id",
        warn: Callable[[str], None] = print,
    ) -> Dict[str, Dict[str, Any]]:
        """Ensure rows keyed by `key` are unique, discarding duplicates and missing entries."""
        out: Dict[str, Dict[str, Any]] = {}
        for i, row in enumerate(rows):
            if key not in row:
                warn(f"WARNING: row {i} missing '{key}'; skipping.")
                continue
            k = row[key]
            if k in out:
                warn(f"WARNING: duplicate {key}={k} at row {i}; discarding later occurrence.")
                continue
            out[k] = row
        return out

    @staticmethod
    def iter_batches(seq: List[Any], batch_size: int) -> Iterable[List[Any]]:
        """Yield sequential slices of `seq` capped at `batch_size`."""
        for i in range(0, len(seq), batch_size):
            yield seq[i:i + batch_size]

    # --- public API -------------------------------------------------------
    def fetch_availability(
        self,
        property_ids: Union[str, List[str]],
        checkin: str,
        checkout: str,
        occupancy: Union[int, str],
        rate_type: str,
        *,
        rate_plan_count: int = 1,
        currency: str = "USD",
        country_code: str = "US",
        language: str = "en-US",
        sales_channel: str = "website",
        sales_environment: str = "hotel_package",
        customer_ip: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return availability responses for up to 250 properties in a single call."""
        ids = [property_ids] if isinstance(property_ids, str) else list(property_ids)
        if len(ids) > 250:
            raise ValueError("Only 250 properties allowed per availability request.")

        if rate_type not in self.RATE_PRESETS:
            raise ValueError(f"Unknown rate_type '{rate_type}'. Valid keys: {sorted(self.RATE_PRESETS)}")

        rate_config = self.RATE_PRESETS[rate_type]
        params: Dict[str, Any] = {
            "checkin": checkin,
            "checkout": checkout,
            "currency": currency,
            "country_code": country_code,
            "language": language,
            "occupancy": str(occupancy),
            "property_id": ids,
            "rate_plan_count": rate_plan_count,
            "sales_channel": sales_channel,
            "sales_environment": sales_environment,
            "payment_terms": rate_config["payment_terms"],
            "partner_point_of_sale": rate_config["partner_point_of_sale"],
        }

        data = self._request(
            "get",
            "v3/properties/availability",
            params=params,
            customer_ip=customer_ip or self.customer_ip,
        )

        if isinstance(data, list):
            return self._to_unique_dict(data)
        if isinstance(data, dict):
            return data
        raise ExpediaAPIError("Unexpected availability payload format.")

    def fetch_property_summaries(
        self,
        property_ids: List[str],
        *,
        includes: Iterable[str] = ("name", "address", "location", "statistics"),
        language: str = "en-US",
        supply_source: str = "expedia",
    ) -> Dict[str, Any]:
        """Fetch property content details for the requested property IDs."""
        if not property_ids:
            return {}
        if len(property_ids) > 250:
            raise ValueError("Only 250 property IDs allowed per content request.")

        params: List[Tuple[str, Any]] = [("language", language), ("supply_source", supply_source)]
        for inc in includes:
            params.append(("include", inc))
        for pid in property_ids:
            params.append(("property_id", pid))

        data = self._request(
            "get",
            "v3/properties/content",
            params=params,
            customer_ip=self.customer_ip,
        )

        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            out: Dict[str, Any] = {}
            for item in data:
                if isinstance(item, dict) and "property_id" in item:
                    out[str(item["property_id"])] = item
            return out
        raise ExpediaAPIError("Unexpected property content payload format.")

    def search_geography(
        self,
        polygon_geojson: Dict[str, Any],
        *,
        include: str = "property_ids",
        supply_source: Optional[str] = "expedia",
        customer_session_id: Optional[str] = None,
        endpoint_path: str = "v3/properties/geography",
        timeout: Optional[int] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
    ) -> List[str]:
        """Retrieve property IDs matching a polygon/shape search request."""
        if checkin:
            try:
                checkin_date = datetime.strptime(checkin, "%Y-%m-%d").date()
            except ValueError as exc:  # invalid formatted input
                raise ExpediaAPIError("search_geography checkin must be YYYY-MM-DD.") from exc
            if checkin_date < date.today():
                raise ExpediaAPIError("search_geography does not support stay dates before today.")

        if checkout:
            try:
                checkout_date = datetime.strptime(checkout, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ExpediaAPIError("search_geography checkout must be YYYY-MM-DD.") from exc
            if checkout_date < date.today():
                raise ExpediaAPIError("search_geography does not support stay dates before today.")

        params: Dict[str, Any] = {"include": include}
        if supply_source:
            params["supply_source"] = supply_source

        extra_headers: Dict[str, str] = {"User-Agent": "TravelNow/3.30.112"}
        if customer_session_id:
            extra_headers["Customer-Session-Id"] = customer_session_id

        data = self._request(
            "post",
            endpoint_path,
            params=params,
            json_payload=polygon_geojson,
            extra_headers=extra_headers,
            timeout=timeout,
        )

        if isinstance(data, dict):
            prop_ids = set()
            for k, v in data.items():
                if isinstance(k, str):
                    prop_ids.add(k)
                if isinstance(v, dict) and "property_id" in v:
                    prop_ids.add(str(v["property_id"]))
            return sorted(prop_ids)
        if isinstance(data, list):
            prop_ids = {
                str(item["property_id"])
                for item in data
                if isinstance(item, dict) and "property_id" in item
            }
            return sorted(prop_ids)
        raise ExpediaAPIError("Unexpected geography payload format.")

    def region_properties_near(
        self,
        lat: float,
        lon: float,
        *,
        radius_km: int = 1,
        limit: int = 100,
        language: str = "en-US",
        supply_source: str = "expedia",
        customer_ip: Optional[str] = None,
    ) -> List[str]:
        """Return property IDs near the given lat/lon within the provided radius."""
        params = {
            "area": f"{int(radius_km)},{lat},{lon}",
            "include": "property_ids",
            "limit": limit,
            "language": language,
            "supply_source": supply_source,
        }
        data = self._request(
            "get",
            "v3/regions",
            params=params,
            customer_ip=customer_ip or self.customer_ip,
        )

        if isinstance(data, list):
            property_ids: List[str] = []
            for region in data:
                for p_id in region.get("property_ids", {}) or []:
                    property_ids.append(p_id)
            return property_ids
        if isinstance(data, dict):
            property_ids = []
            for region in data.values():
                if isinstance(region, dict):
                    for p_id in region.get("property_ids", {}) or []:
                        property_ids.append(p_id)
            return property_ids
        raise ExpediaAPIError("Unexpected regions payload format.")

    def run_batched(
        self,
        ids: List[str],
        jobs: Iterable[Tuple[Callable[[List[str]], Mapping[Any, Any]], Dict[Any, Any]]],
        *,
        batch_size: int = 250,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Execute batch jobs across IDs, merging mapping results into the provided sinks."""
        batches = list(self.iter_batches(ids, batch_size))
        total = len(batches)
        for idx, batch in enumerate(batches, start=1):
            if on_progress:
                on_progress(idx, total)
            for call, sink in jobs:
                result = call(batch)
                if result is None:
                    continue
                if not isinstance(result, Mapping):
                    raise ExpediaAPIError("Batched job callables must return mapping-like responses.")
                sink.update(result)

@lru_cache(maxsize=1)
def _get_default_client() -> ExpediaClient:
    """Instantiate (and memoise) a default client using environment variables."""
    return ExpediaClient.from_env()


def auth_env():
    """Return a tuple of (authorization header, base URL) using the cached client."""
    client = _get_default_client()
    return auth_header(client.api_key, client.shared_secret), client.base_url
