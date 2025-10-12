import requests
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class APIRequestHandler:
    def __init__(self, base_url: str, token: Optional[str] = None):
        if not base_url.endswith('/'):
            base_url += '/'
        self.base_url = base_url
        self.session = requests.Session()
        
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _send_request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        url = self.base_url + endpoint
        try:
            logging.info(f"Sending {method} request to {url} with kwargs: {kwargs}")
            response = self.session.request(method, url, **kwargs, timeout=10)
            response.raise_for_status()

            return response
        
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error for {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection Error for {url}: {e}")
        except requests.exceptions.Timeout as e:
            logging.error(f"Timeout Error for {url}: {e}")
        except requests.exceptions.RequestException as e:
            logging.error(f"An unexpected error occurred for {url}: {e}")
        
        return requests.Response()
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None)  -> requests.Response:
        """Sends a GET request."""
        return self._send_request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Dict[str, Any]) -> requests.Response:
        """Sends a POST request with a JSON payload."""
        return self._send_request("POST", endpoint, json=data)

    def put(self, endpoint: str, data: Dict[str, Any]) -> requests.Response:
        """Sends a PUT request with a JSON payload."""
        return self._send_request("PUT", endpoint, json=data)

    def delete(self, endpoint: str) -> requests.Response:
        """Sends a DELETE request."""
        return self._send_request("DELETE", endpoint)

    def close_session(self):
        """Closes the session."""
        self.session.close()
        logging.info("HTTP session closed.")