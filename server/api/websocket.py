"""
Raw WebSocket support for the Zappy server (REST + WS on same Flask app).
Implements native WebSocket at /ws using flask-sock, compatible with the GUI
which uses the browser WebSocket API (not Socket.IO).
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Set

from flask_sock import Sock
from server.logging_config import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Manage raw WebSocket connections and broadcast server events."""

    def __init__(self, app, server, api):
        self.app = app
        self.server = server
        self.api = api
        self.sock = Sock(app)
        self._clients: Set[object] = set()
        self._lock = threading.Lock()

        # Register WS route
        self._register_routes()
        logger.info("WebSocket route registered at /ws")
        # Hook API events to push updates to clients
        self._register_api_callbacks()
        logger.debug("WebSocket callbacks registered")

    def _register_routes(self):
        @self.sock.route('/ws')
        def ws_route(ws):
            # Add to connected clients
            with self._registered(ws):
                logger.info("WebSocket client connected | total=%s", self._client_count)
                # Send initial connection info
                try:
                    ws.send(json.dumps({
                        'type': 'connected',
                        'data': {
                            'ticks': getattr(self.server, 'ticks', 0),
                            'map_size': getattr(self.server.map, 'size', 0),
                            'teams': list(getattr(self.server, 'teams', [])),
                            'players': len(getattr(self.server, 'players', [])),
                        }
                    }))
                except Exception as e:
                    logger.exception("Failed sending initial WS payload: %s", e)
                    return

                # Read loop (optional): currently just consume messages and ignore
                while True:
                    try:
                        msg = ws.receive()  # Blocks until message or close
                        if msg is None:
                            logger.info("WebSocket client disconnected | total=%s", self._client_count)
                            break  # Client closed
                        # Optionally handle client messages in future
                        # For now, support a simple ping
                        try:
                            payload = json.loads(msg)
                            if isinstance(payload, dict) and payload.get('type') == 'ping':
                                ws.send(json.dumps({'type': 'pong', 'data': {}}))
                                logger.debug("WS ping -> pong")
                            else:
                                logger.debug("WS client message ignored: %s", msg)
                        except Exception:
                            # Non-JSON or unsupported message; ignore
                            logger.debug("WS non-JSON or invalid message: %s", msg)
                            pass
                    except Exception as e:
                        logger.warning("WebSocket receive loop error: %s", e)
                        break

    def _register_api_callbacks(self):
        # Player property change -> player_update
        self.api.register_callback('player_property_changed', lambda d: self._broadcast_safe({
            'type': 'player_update',
            'data': self.api.get_player_data(d['player_id'])
        }))
        # Player death -> player_update
        self.api.register_callback('player_died', lambda d: self._broadcast_safe({
            'type': 'player_update',
            'data': d
        }))
        # Step end -> server_update (stats)
        self.api.register_callback('step_end', lambda d: self._broadcast_safe({
            'type': 'server_update',
            'data': self.api.get_statistics()
        }))
        # Tile changed -> map_update
        self.api.register_callback('tile_changed', lambda d: self._broadcast_safe({
            'type': 'map_update',
            'data': d
        }))
        # Team add/remove -> server_update summary
        self.api.register_callback('team_added', lambda d: self._broadcast_safe({
            'type': 'server_update',
            'data': self.api.get_statistics()
        }))
        self.api.register_callback('team_removed', lambda d: self._broadcast_safe({
            'type': 'server_update',
            'data': self.api.get_statistics()
        }))

    @contextmanager
    def _registered(self, ws):
        self._add_client(ws)
        try:
            yield
        finally:
            self._remove_client(ws)

    def _add_client(self, ws):
        with self._lock:
            self._clients.add(ws)
            logger.debug("WS client added | total=%s", len(self._clients))

    def _remove_client(self, ws):
        with self._lock:
            self._clients.discard(ws)
            logger.debug("WS client removed | total=%s", len(self._clients))

    @property
    def _client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _broadcast_safe(self, message: dict):
        data = json.dumps(message)
        dead = []
        with self._lock:
            for ws in list(self._clients):
                try:
                    ws.send(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)
        logger.debug("WS broadcast | type=%s clients=%s dead=%s", message.get('type'), self._client_count, len(dead))
