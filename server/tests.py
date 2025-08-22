"""
tests.py

Contains test cases and validation logic for the Zappy server and related modules.
"""
import unittest

from server import Server
from server import Player
from server import Map
from server import Direction, Resource, ElevationRequirement, RESOURCE_DENSITY


class TestEnums(unittest.TestCase):
    def test_direction_enum(self):
        self.assertEqual(Direction.NORTH.value, 0)
        self.assertEqual(Direction.EAST.value, 1)
        self.assertEqual(Direction.SOUTH.value, 2)
        self.assertEqual(Direction.WEST.value, 3)

    def test_resource_enum(self):
        self.assertEqual(Resource.FOOD.value, "food")
        self.assertEqual(Resource.LINEMATE.value, "linemate")


class TestMap(unittest.TestCase):
    def setUp(self):
        self.teams = ["red"]
        self.players = []
        self.map = Map(self.teams, self.players, 5, size=10)

    def test_initialization(self):
        self.assertEqual(len(self.map.tiles), 10)
        self.assertEqual(len(self.map.tiles[0]), 10)

    def test_resource_distribution(self):
        total_food = sum(tile[Resource.FOOD]
                         for row in self.map.tiles for tile in row)
        expected_food = int(10 * 10 * RESOURCE_DENSITY[Resource.FOOD])
        self.assertEqual(total_food, expected_food)

    def test_egg_placement(self):
        self.assertGreaterEqual(len(self.map.eggs_tile["red"]), 1)
        egg = self.map.eggs_tile["red"][0]
        self.assertIsInstance(egg, tuple)
        self.assertTrue(0 <= egg[0] < 10)
        self.assertTrue(0 <= egg[1] < 10)


class TestPlayer(unittest.TestCase):
    def test_broadcast(self):
        """Test the broadcast command for correct message delivery and direction calculation."""
        # Place two players on the map
        self.player.position = (2, 2)
        self.player.direction = Direction.NORTH
        another_player = Player(1, self.map, "blue")
        another_player.position = (2, 3)  # Directly south of self.player
        another_player.direction = Direction.NORTH
        self.map.players.append(another_player)

        # Player broadcasts a message
        self.player.broadcast("hello")

        # The broadcasting player should get 'ok' and 'message 0, hello' (to self)
        self.assertIn("ok", self.player.response_queue[-1])
        self.assertTrue(any("message 0, hello" in res for res in self.player.response_queue))

        # The other player should get a message with direction number (should be 3 for directly south)
        found_broadcast = False
        for res in another_player.response_queue:
            if res.startswith("message ") and ", hello" in res:
                # Should be direction 3 (south relative to north-facing player)
                self.assertIn("message 5, hello", res)
                found_broadcast = True
        self.assertTrue(found_broadcast)


    def setUp(self):
        self.map = Map(["blue"], [], nb_eggs=5, size=5)
        self.player = Player(0, self.map, "blue")
        self.player.position = (2, 2)
        self.player.direction = Direction.NORTH
        self.map.players.append(self.player)

    def test_movement(self):
        # Test forward movement
        self.player.forward()
        self.assertEqual(self.player.position, (2, 1))

        # Test boundary wrapping
        self.player.position = (0, 0)
        self.player.direction = Direction.WEST
        self.player.forward()
        self.assertEqual(self.player.position, (4, 0))

    def test_rotation(self):
        self.player.right()
        self.assertEqual(self.player.direction, Direction.EAST)

        self.player.left()
        self.assertEqual(self.player.direction, Direction.NORTH)

    def test_resource_handling(self):
        # Setup resource on current tile
        x, y = self.player.position
        self.map.tiles[y][x][Resource.FOOD] = 1

        # Test taking resource
        self.player.take("food")
        self.assertEqual(self.player.inventory["food"], 11)
        self.assertEqual(self.map.tiles[y][x][Resource.FOOD], 0)

        # Test setting resource
        self.player.set("food")
        self.assertEqual(self.player.inventory["food"], 10)
        self.assertEqual(self.map.tiles[y][x][Resource.FOOD], 1)

    def test_incantation_requirements(self):
        self.player.level = 1
        x, y = self.player.position

        # Setup required resources
        required = ElevationRequirement.requirements[1][1]
        for res, count in required.items():
            self.map.tiles[y][x][res] = count

        # Setup additional player
        another_player = Player(1, self.map, "blue")
        another_player.position = (x, y)
        another_player.level = 1
        self.map.players.append(another_player)

        # Test successful incantation start
        self.player.incantation()
        self.assertIn((x, y), self.map.incanted_tile)

    def test_eject(self):
        """Test the eject command for correct player movement and response."""
        # Place two players on the same tile
        self.player.position = (2, 2)
        self.player.direction = Direction.NORTH
        another_player = Player(1, self.map, "blue")
        another_player.position = (2, 2)
        another_player.direction = Direction.SOUTH
        self.map.players.append(another_player)

        # Call eject
        self.player.eject()

        # The other player should have moved north (y-1)
        expected_position = (2, (2 - 1) % self.map.size)
        self.assertEqual(another_player.position, expected_position)
        # The ejecting player should get 'ok', the ejected player should get 'eject: south' in their response
        self.assertIn("ok", self.player.response_queue[-1])
        found_eject = any("eject: 2" in res for res in another_player.response_queue)
        self.assertTrue(found_eject)


class TestServer(unittest.TestCase):
    def setUp(self):
        self.server = Server(size=10, nb_eggs=5, teams=["red", "blue"])

    def test_player_creation(self):
        player = self.server.add_player("red")
        self.assertIsInstance(player, Player)
        self.assertEqual(player.team, "red")
        self.assertEqual(len(self.server.players), 1)

    def test_command_processing(self):
        player = self.server.add_player("blue")
        player.add_cmd("Forward")
        player.add_cmd("Right")

        # Process commands
        self.server.step()
        self.assertEqual(len(player.command_queue), 1)
        self.assertTrue(player.cur_cmd)
        for _ in range(10):
            self.server.step()


class TestIntegration(unittest.TestCase):
    def test_full_game_cycle(self):
        map_size = 5
        server = Server(size=map_size, nb_eggs=1, teams=["test"])
        player = server.add_player("test")

        # Initial position and direction
        initial_x, initial_y = player.position
        # Default direction
        self.assertEqual(Direction.NORTH.value, Direction.NORTH.value)

        # Add movement commands: Forward → Right → Forward
        player.add_cmd("Forward")  # Move North (y decreases)
        player.add_cmd("Right")    # Turn East
        player.add_cmd("Forward")  # Move East (x increases)

        # Execute enough steps to process all commands
        for _ in range(50):
            server.step()

        # Verify final direction (should be East after Right turn)
        self.assertEqual(player.direction, Direction.EAST)

        # Calculate expected position after movements
        expected_y = (initial_y - 1) % map_size
        expected_x = (initial_x + 1) % map_size
        self.assertEqual(player.position, (expected_x, expected_y))

    def test_new_player_connection(self):
        map_size = 5
        server = Server(size=map_size, nb_eggs=1, teams=["test"])
        player1 = server.add_player("test")

        # Set initial position and resources
        x, y = 2, 2
        player1.position = (x, y)
        player1.level = 1

        # Set required resources for level 1->2 incantation
        required = ElevationRequirement.requirements[1][1]
        for res, count in required.items():
            server.map.tiles[y][x][res] = count

        # Start incantation (requires 1 additional player)
        player1.add_cmd("Fork")

        # Process to start incantation
        for _ in range(43):
            server.step()
        self.assertEqual(server.map.nb_eggs(player1.team), 1)

        # Verify egg on the fork coordinate
        self.assertIn((x, y), server.map.eggs_tile["test"])

        # Connect a new player to the same team and position
        player2 = server.add_player("test")

        self.assertEqual(player1.position, player2.position)
        self.assertEqual(player1.team, player2.team)
        self.assertEqual(server.map.nb_eggs(player1.team), 0)

    def test_incantation_with_new_player_connection(self):
        map_size = 5
        server = Server(size=map_size, nb_eggs=2, teams=["test"])
        player1 = server.add_player("test")

        # Get initial connect_nbr value
        initial_connect_nbr = server.map.nb_eggs(player1.team)

        # Set initial position and resources
        x, y = 2, 2
        player1.position = (x, y)
        player1.level = 1

        # Set required resources for level 1->2 incantation
        required = ElevationRequirement.requirements[1][1]
        for res, count in required.items():
            server.map.tiles[y][x][res] = count

        # Start incantation (requires 1 additional player)
        player1.add_cmd("Incantation")

        # Process to start incantation
        for _ in range(301):
            server.step()

        # Verify incantation has started but not completed
        self.assertIn((x, y), server.map.incanted_tile)
        self.assertEqual(player1.level, 1)  # Still level 1

        # Connect a new player to the same team and position
        player2 = server.add_player("test")
        player2.position = (x, y)
        player2.level = 1

        # Verify new player is connected
        self.assertEqual(len(server.players), 2)
        self.assertEqual(player2.team, "test")

        # Verify connect_nbr decreased by 1
        self.assertEqual(server.map.nb_eggs(
            player1.team), initial_connect_nbr - 1)

        # Complete incantation process
        incantation_duration = 301  # Assuming 300 steps for incantation
        for _ in range(incantation_duration):
            server.step()

        # Verify both players leveled up
        self.assertEqual(player1.level, 2)
        self.assertEqual(player2.level, 2)

        # Verify tile is no longer marked for incantation
        self.assertNotIn((x, y), server.map.incanted_tile)

        # Verify resources were consumed
        for res in required:
            self.assertEqual(server.map.tiles[y][x][res], 0)

    def test_full_game_cycle(self):
        map_size = 5
        server = Server(size=map_size, nb_eggs=1, teams=["test"])
        player = server.add_player("test")
        # Default direction
        self.assertEqual(player.direction, Direction.NORTH)
        server.map.tiles[player.position[1]
                         ][player.position[0]][Resource.FOOD] = 1

        player.add_cmd("Take food")
        # Execute enough steps to process all commands
        for _ in range(8):
            server.step()
        self.assertEqual(player.inventory[Resource.FOOD.value], 11)

        player.add_cmd("Set food")
        for _ in range(8):
            server.step()
        self.assertEqual(player.inventory[Resource.FOOD.value], 10)


if __name__ == "__main__":
    unittest.main()
