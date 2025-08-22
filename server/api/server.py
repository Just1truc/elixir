"""
_single_port_server.py

 API server that runs both REST API and WebSocket on the same port
for the Zappy server simulation.

This module provides:
1. REST API endpoints at /api/* (HTTP)
2. WebSocket API at /ws (WebSocket protocol)
3. Both running on the same port with Flask-SocketIO

Benefits:
- Single port deployment
- Easier firewall configuration
- Simplified client connection logic
- Shared session management
"""

from server.api.rest import ZappyRestAPI
from server import Server
from server.api.websocket import WebSocketManager
from server.logging_config import configure_logging, get_logger
import argparse

logger = get_logger(__name__)


class ZappyAPIServer:
    """Server that runs REST API and raw WebSocket on the same app."""
    
    def __init__(self, server_size=10, nb_eggs=5, teams=None, server=None):
        """
        Initialize server.
        
        Args:
            server_size: Map size (if creating new server)
            nb_eggs: Eggs per team (if creating new server)
            teams: Team names (if creating new server)
            server: Existing server instance (optional)
        """
        if teams is None:
            teams = ["team1", "team2"]
        
        # Create or use existing server
        if server is not None:
            self.server = server
        else:
            self.server = Server(server_size, nb_eggs, teams)
            # Add initial players
            for team in teams:
                self.server.add_player(team)
        
        # Create REST API and reuse its Flask app
        self.rest_api = ZappyRestAPI(server=self.server)
        self.app = self.rest_api.app
        self.api = self.rest_api.api

        # Attach raw WebSocket manager (flask-sock) on the same app
        self.ws = WebSocketManager(self.app, self.server, self.api)
        logger.info("API server initialized | size=%s eggs=%s teams=%s", server_size, nb_eggs, teams)
    
    def run(self, host='localhost', port=5000, debug=False):
        """
        Run the REST + WS server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        logger.info("Starting Zappy API on %s:%s (debug=%s)", host, port, debug)
        logger.info("REST: http://%s:%s/api/health | Docs: /api/docs | WS: ws://%s:%s/ws", host, port, host, port)
        # Note: Werkzeug dev server supports WS for development via flask-sock.
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the API server."""
    configure_logging()
    parser = argparse.ArgumentParser(description='Zappy Server API (REST + WebSocket)')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--size', type=int, default=10, help='Map size')
    parser.add_argument('--eggs', type=int, default=5, help='Number of eggs')
    parser.add_argument('--teams', nargs='+', default=['team1', 'team2'], help='Team names')
    args = parser.parse_args()
    
    srv = ZappyAPIServer(server_size=args.size, nb_eggs=args.eggs, teams=args.teams)
    srv.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
