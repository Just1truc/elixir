from server import Server


main = Server(42, 5, ["red", "blue"])
# ai = ai.Ai(player)
player = main.add_player("red")

for i in range(80):
    player.add_cmd("Forward")
    main.step()
    # ai.step()

    res = player.get_res()
    if res:
        print(res)
    if player.is_alive == False:
        print("player is dead")
