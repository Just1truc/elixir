"""
server.py

Flask-based REST API for the Zappy server simulation.
Provides HTTP endpoints for all server data access and manipulation capabilities.

The REST API can be used in two ways:
1. Standalone mode: Creates its own server instance
2. Integration mode: Uses an existing server instance

Endpoints:
- GET /api/server - Get complete server data
- GET /api/server/stats - Get server statistics
- GET /api/players - Get all players
- GET /api/players/{id} - Get specific player
- PUT /api/players/{id} - Update player property
- GET /api/teams - Get all teams
- GET /api/teams/{name} - Get specific team data
- POST /api/teams - Add new team
- DELETE /api/teams/{name} - Remove team
- GET /api/map - Get map data
- PUT /api/map/tile/{x}/{y} - Update map tile
- POST /api/search/players - Search players with criteria

Usage:
    # Standalone mode (creates new server)
    python -m api.rest.server
    
    # Integration mode (use existing server)
    from server import Server
    from api.rest import create_rest_api_for_server
    
    server = Server(10, 5, ["team1", "team2"])
    # ... configure server ...
    api = create_rest_api_for_server(server)
    api.run()
    
    # In another terminal:
    curl http://localhost:5000/api/server/stats
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import time

from server import Server
from server.api.core import ServerAPI
from server.logging_config import get_logger

logger = get_logger(__name__)


class ZappyRestAPI:
    """REST API wrapper for Zappy server using Flask."""
    
    def __init__(self, server=None, server_size=10, nb_eggs=5, teams=None):
        """
        Initialize the REST API with a Zappy server.
        """
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web applications
        logger.info("Initializing REST API | size=%s eggs=%s teams=%s provided_server=%s", server_size, nb_eggs, teams, server is not None)
        
        # Use provided server or create new one
        if server is not None:
            self.server = server
            if not hasattr(self.server, 'api') or self.server.api is None:
                self.api = ServerAPI(self.server)
                self.server.api = self.api
            else:
                self.api = self.server.api
        else:
            if teams is None:
                teams = ["team1", "team2"]
            self.server = Server(server_size, nb_eggs, teams)
            self.api = ServerAPI(self.server)
            self.server.api = self.api
            for team in teams:
                self.server.add_player(team)
        
        # Setup routes
        self.setup_routes()
        logger.info("REST API ready with routes registered")
    
    def setup_routes(self):
        """Setup all Flask routes."""
        
        # Health/Docs endpoints
        @self.app.route('/api/health', methods=['GET'])
        def health():
            """Lightweight health check."""
            try:
                return jsonify({
                    'success': True,
                    'status': 'ok',
                    'ticks': getattr(self.server, 'ticks', 0),
                    'players': len(getattr(self.server, 'players', [])),
                    'teams': len(getattr(self.server, 'teams', []))
                })
            except Exception as e:
                logger.exception("Health check failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/docs', methods=['GET'])
        def docs():
            """Very simple docs/help page for the REST API."""
            return (
                """
                <html>
                  <head><title>Zappy REST API</title></head>
                  <body>
                    <h1>Zappy REST API</h1>
                    <ul>
                      <li>GET /api/health</li>
                      <li>GET /api/server</li>
                      <li>GET /api/server/stats</li>
                      <li>POST /api/server/step</li>
                      <li>POST /api/server/reset</li>
                      <li>GET /api/players?team=NAME&alive=true</li>
                      <li>GET /api/players/&lt;id&gt;</li>
                      <li>PUT /api/players/&lt;id&gt; {"property": "level", "value": 2}</li>
                      <li>GET /api/teams</li>
                      <li>GET /api/teams/&lt;name&gt;</li>
                      <li>POST /api/teams {"name": "team3"}</li>
                      <li>DELETE /api/teams/&lt;name&gt;</li>
                      <li>GET /api/map</li>
                      <li>PUT /api/map/tile/&lt;x&gt;/&lt;y&gt; {"resource": "food", "amount": 3} or {"food":3,"linemate":1,...}</li>
                      <li>POST /api/search/players {"team": "red"}</li>
                    </ul>
                  </body>
                </html>
                """,
                200,
                {'Content-Type': 'text/html'}
            )
        
        # Server endpoints
        @self.app.route('/api/server', methods=['GET'])
        def get_server_data():
            try:
                data = self.api.get_server_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                logger.exception("/api/server failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/server/stats', methods=['GET'])
        def get_server_stats():
            try:
                stats = self.api.get_statistics()
                return jsonify({'success': True, 'data': stats})
            except Exception as e:
                logger.exception("/api/server/stats failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/server/step', methods=['POST'])
        def step_server():
            try:
                self.server.step()
                return jsonify({'success': True, 'data': {'tick': self.server.ticks, 'message': 'Server stepped successfully'}})
            except Exception as e:
                logger.exception("/api/server/step failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/server/reset', methods=['POST'])
        def reset_server():
            try:
                payload = request.get_json(silent=True) or {}
                nb_eggs = payload.get('nb_eggs')
                teams = payload.get('teams')
                if teams is not None:
                    self.server.teams = list(teams)
                    if hasattr(self.server, 'map') and hasattr(self.server.map, 'teams'):
                        self.server.map.teams = list(teams)
                self.server.reset(nb_eggs=nb_eggs, teams=teams)
                for team in self.server.teams:
                    self.server.add_player(team)
                return jsonify({'success': True, 'data': {'message': 'Server reset successfully', 'ticks': self.server.ticks, 'teams': self.server.teams}})
            except Exception as e:
                logger.exception("/api/server/reset failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Player endpoints
        @self.app.route('/api/players', methods=['GET'])
        def get_all_players():
            try:
                team = request.args.get('team')
                alive_only = request.args.get('alive') == 'true'
                if team:
                    players = self.api.get_players_by_team(team)
                else:
                    server_data = self.api.get_server_data()
                    players = server_data['players']
                if alive_only:
                    players = [p for p in players if p['is_alive']]
                return jsonify({'success': True, 'data': players, 'count': len(players)})
            except Exception as e:
                logger.exception("/api/players failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/players/<int:player_id>', methods=['GET'])
        def get_player(player_id):
            try:
                player = self.api.get_player_data(player_id)
                if player is None:
                    return jsonify({'success': False, 'error': 'Player not found'}), 404
                return jsonify({'success': True, 'data': player})
            except Exception as e:
                logger.exception("/api/players/<id> failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/players/<int:player_id>', methods=['PUT'])
        def update_player(player_id):
            try:
                data = request.get_json()
                if not data or 'property' not in data or 'value' not in data:
                    return jsonify({'success': False, 'error': 'Missing property or value in request body'}), 400
                success = self.api.set_player_property(player_id, data['property'], data['value'])
                if not success:
                    return jsonify({'success': False, 'error': 'Failed to update player property'}), 400
                return jsonify({'success': True, 'data': {'message': f"Updated player {player_id} {data['property']} to {data['value']}"}})
            except Exception as e:
                logger.exception("/api/players PUT failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Team endpoints
        @self.app.route('/api/teams', methods=['GET'])
        def get_all_teams():
            try:
                teams_data = []
                for team in self.server.teams:
                    team_data = self.api.get_team_data(team)
                    teams_data.append(team_data)
                return jsonify({'success': True, 'data': teams_data, 'count': len(teams_data)})
            except Exception as e:
                logger.exception("/api/teams failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/teams/<string:team_name>', methods=['GET'])
        def get_team(team_name):
            try:
                if team_name not in self.server.teams:
                    return jsonify({'success': False, 'error': 'Team not found'}), 404
                team_data = self.api.get_team_data(team_name)
                return jsonify({'success': True, 'data': team_data})
            except Exception as e:
                logger.exception("/api/teams/<name> failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/teams', methods=['POST'])
        def add_team():
            try:
                data = request.get_json()
                if not data or 'name' not in data:
                    return jsonify({'success': False, 'error': 'Missing team name in request body'}), 400
                success = self.api.add_team(data['name'])
                if not success:
                    return jsonify({'success': False, 'error': 'Team already exists'}), 400
                return jsonify({'success': True, 'data': {'message': f"Team {data['name']} added successfully"}}), 201
            except Exception as e:
                logger.exception("/api/teams POST failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/teams/<string:team_name>', methods=['DELETE'])
        def remove_team(team_name):
            try:
                success = self.api.remove_team(team_name)
                if not success:
                    return jsonify({'success': False, 'error': 'Team not found'}), 404
                return jsonify({'success': True, 'data': {'message': f"Team {team_name} removed successfully"}})
            except Exception as e:
                logger.exception("/api/teams DELETE failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Map endpoints
        @self.app.route('/api/map', methods=['GET'])
        def get_map_data():
            try:
                data = self.api.get_map_data()
                return jsonify({'success': True, 'data': data})
            except Exception as e:
                logger.exception("/api/map failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/map/tile/<int:x>/<int:y>', methods=['PUT'])
        def update_map_tile(x, y):
            """Update a specific map tile's resource amount. Accepts either a single resource or a dict of resources."""
            try:
                data = request.get_json() or {}
                # Single resource update
                if 'resource' in data and 'amount' in data:
                    success = self.api.set_map_tile(x, y, data['resource'], data['amount'])
                    if not success:
                        return jsonify({'success': False, 'error': 'Failed to update map tile'}), 400
                else:
                    # Bulk update: iterate over provided resources
                    any_success = False
                    for res, amt in data.items():
                        if self.api.set_map_tile(x, y, res, amt):
                            any_success = True
                    if not any_success:
                        return jsonify({'success': False, 'error': 'No resources updated'}), 400
                return jsonify({'success': True, 'data': {'message': 'Tile updated'}})
            except Exception as e:
                logger.exception("/api/map/tile PUT failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # Search endpoints
        @self.app.route('/api/search/players', methods=['POST'])
        def search_players():
            try:
                criteria = request.get_json() or {}
                results = self.api.search_players(**criteria)
                return jsonify({'success': True, 'data': results, 'count': len(results)})
            except Exception as e:
                logger.exception("/api/search/players failed: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500


def create_rest_api_for_server(server: Server) -> ZappyRestAPI:
    """Create a REST API instance for an existing server."""
    return ZappyRestAPI(server=server)


def main():
    """Run REST API as a standalone server for development purposes."""
    from server.logging_config import configure_logging
    configure_logging()
    api = ZappyRestAPI()
    api.app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
