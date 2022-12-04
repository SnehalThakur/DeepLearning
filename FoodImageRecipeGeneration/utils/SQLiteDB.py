import sqlite3 as sql


def createTableIfNotExist():
    sqlConnection = sql.connect(r"SQLiteDB\\recipeData.db")
    print(sqlConnection)

    sqlConnection.execute("""
                        CREATE TABLE IF NOT EXISTS recipe (
                            id integer primary key autoincrement,
                            itemName text not null,
                            recipeDescription text not null,
                            ingredients text not null
                        );
                    """)


# cursor = sqlConnection.execute("""SELECT name FROM sqlite_master WHERE type='table';""")
# print(cursor.fetchall())

def insertRecipeData(itemName, recipeDescription, ingredients):
    con = sql.connect("SQLiteDB\\recipeData.db")
    cur = con.cursor()
    cur.execute("INSERT INTO recipe (itemName,recipeDescription,ingredients) VALUES (?,?,?)",
                (itemName, recipeDescription, ingredients))
    con.commit()
    con.close()


def retrieveRecipeData():
    con = sql.connect("SQLiteDB\\recipeData.db")
    cur = con.cursor()
    cur.execute("SELECT itemName,recipeDescription,ingredients FROM recipe")
    recipe = cur.fetchall()
    con.close()
    return recipe


def retrieveRecipeDataWithItemName(itemName):
    con = sql.connect("SQLiteDB\\recipeData.db")
    cur = con.cursor()
    cur.execute(f"SELECT * FROM recipe WHERE itemName='{itemName}'")
    recipe = cur.fetchall()
    con.close()
    return recipe

# createTableIfNotExist()
# # apple_pie  recipe
# insertRecipeData("apple_pie", "Step 1: Peel and core apples, then thinly slice and Set aside; Step 2: Preheat the oven to 425 degrees F (220 degrees C). Step 3: Melt butter in a saucepan over medium heat. Add flour and stir to form a paste; cook until fragrant, about 1 to 2 minutes. Add both sugars and water; bring to a boil; Reduce the heat to low and simmer for 3 to 5 minutes; Remove from the heat. Step 4: Press one pastry into the bottom and up the sides of a 9-inch pie pan; Roll out remaining pastry so it will overhang the pie by about 1/2 inch; Cut pastry into eight 1-inch strips. Step 5: Place sliced apples into the bottom crust, forming a slight mound. Lay four pastry strips vertically and evenly spaced over apples, using longer strips in the center and shorter strips at the edges. Step 6: Make a lattice crust: Fold the first and third strips all the way back so they're almost falling off the pie; Lay one of the unused strips perpendicularly over the second and fourth strips, then unfold the first and third strips back into their original position. Step 7: Fold the second and fourth vertical strips back; Lay one of the three unused strips perpendicularly over top; Unfold the second and fourth strips back into their original position. Step 8: Repeat Steps 6 and 7 to weave in the last two strips of pastry; Fold and trim excess dough at the edges as necessary, and pinch to secure. Step 9: Slowly and gently pour sugar-butter mixture over lattice crust, making sure it seeps over sliced apples; Brush some onto lattice, but make sure it doesn't run off the sides. Step 10: Bake in the preheated oven for 15 minutes. Reduce the temperature to 350 degrees F (175 degrees C) and continue baking until apples are soft, 35 to 45 minutes.", "8 small Granny Smith apples, or as needed, ½ cup unsalted butter, 3 tablespoons all-purpose flour, ½ cup white sugar, ½ cup packed brown sugar, ¼ cup water, 1 (9 inch) double-crust pie pastry, thawed")
# # French Fries recipe
# insertRecipeData("french_fries", "Step 1: Chop the potatoes - To prepare this easy recipe, you need to make sure that they are cut in the right shape and size; They should neither be too thick, nor too thin and should be cut clean and sharp; The trick is to first slice the potatoes and then cut them lengthwise. You can also use a fries cutter for that long, even shape. Step 2: Soak cut potatoes in ice-cold water for 10-15 minutes - Now, wash the potatoes under running water till they are squeaky clean; Place them in a bowl of iced water for 10 to 15 minutes; Keep them submerged in water or they will turn black. Step 3: Deep fry the potato fries - Now, heat the oil in a deep bottomed pan; Once the smoke starts appearing, reduce the flame and allow it to acquire a lower temperature; Now, deep fry the potatoes in batches; Keep the flame low; This will make them crunchy and also help retain their colour. Step 4 Sprinkle salt and pepper and serve hot - Drain excess oil and place them on an absorbent paper; Allow them to cool; Now sprinkle salt and pepper and toss well; Serve immediately with ketchup; They can also be served with burgers and cutlets.","500 gm potato, salt as required, 2 cup refined oil")
# # Hamburger recipe
# insertRecipeData("hamburger", "Step 1: Pressure cook carrots, peas and corn - To make the burger patty, pressure cook the carrot, peas and sweet corn for 1 whistle or until soft. Step 2: Add the spices - Add the cooked vegetables, chopped onions, red chilli powder, lemon juice, garam masala powder, salt and ginger-garlic paste to a large bowl. Step 3: Add mashed potatoes - Add lemon juice and mashed potatoes in the bowl, mix well until evenly combined; Shape the mixture into small/medium patties. Step 4: Shallow fry the patties - Now heat oil in a pan over medium flame; Roll the prepared patties in the breadcrumbs and shallow fry until golden brown on both sides. Step 5: Keep aside - Remove and keep aside. Step 6: Prepare the burger - Take one half of the burger bun. Spread some butter and place the lettuce on top. Step 7: Add slices onions, cucumber and tomatoes - Place the prepared vegetable patty on top. Top up with slices of onion, tomato, cucumber and cheese; If you want to make it even more delicious and healthy replace cheese with eggless mayonnaise. Step 8: Burger is ready - Cover it with the other half of the burger bun: If desired, add some ketchup on top; Secure it with a toothpick if desired.","1 sliced onion, 4 slices cheese slices, 1 teaspoon powdered garam masala powder, 2 teaspoon refined oil, 1/2 gm ginger paste, 4 halved burger buns, 2 tablespoon tomato ketchup, 1/2 teaspoon garlic paste")
# # Pizza recipe
# insertRecipeData("pizza", "Step 1: Prepare the pizza dough - Take a dough kneading plate and add all-purpose flour to it; Next, add salt and baking powder in it and sieve the flour once; Then, make a well in the centre and add 1 teaspoon of oil to it. On the other hand, take a little warm water and mix the yeast in it along with 1 teaspoon of sugar; Mix well and keep aside for 10-15 minutes; The yeast will rise in the meantime; Once the yeast has risen, add it to the flour knead the dough nicely using some water; Keep this dough aside for 4-6 hours; Then knead the dough once again; Now, the pizza dough is ready. Step 2: Prepare the pizza base - Preheat the oven at 180 degree Celsius; Now, is the time to make the pizza base when the dough is ready; Dust the space a little using dry flour and take a large amount of the pizza dough; Using a rolling pin, roll this dough into a nice circular base; (Note: Make sure that the circular base is even at all ends) Once you have made the base, use a fork and prick the base with it so that the base doesn't rise and gets baked nicely; Put it into the preheated oven and bake it 10 minutes; Now, your pizza base is ready. Step 3: Chop all the vegetables for the pizza - Now, wash the capsicum and slice it thinly in a bowl; Then, peel the onions and cut thin slices of it as well in another bowl; And finally, cut tomatoes and mushrooms in the same manner; However, make sure that those tomatoes have less juice in them; Once all the veggies are done, Now, grate the processed and mozzarella cheese in separate bowls. Step 4: Spread the sauce and veggies on the base - Then, take the fresh pizza base and apply tomato ketchup all over; Spread half the processed cheese all over the base and evenly put the veggies all across the base; Once you have put all the veggies, put a thick layer of mozzarella cheese. Step 5: Bake the pizza at 250 degree Celsius for 10 minutes - Put this pizza base in a baking tray and place it inside the oven; Let the pizza bake 10 minutes at 250 degree Celsius; Once done, take out the baking tray and slice the pizza; Sprinkle oregano and chilli flakes as per your taste and serve hot; (Note: Make sure that the oven is preheated at 250 degree Celsius for 5 minutes at least)","12 cup all purpose flour, 100 ml tomato ketchup, 1 tomato, 2 onion, 1 teaspoon chilli flakes, 1 teaspoon baking powder, 1 teaspoon sugar, 100 gm processed cheese, 4 mushroom, 1/2 capsicum (green pepper), 1 teaspoon oregano, 1/2 cup mozzarella, 1 tablespoon dry yeast, water as required")
# # Omlet
# insertRecipeData("omelet", "Step 1: BEAT eggs, water, salt and pepper in small bowl until blended. Step 2: HEAT butter in 7 to 10-inch nonstick omelet pan or skillet over medium-high heat until hot. Step 3: TILT pan to coat bottom. Step 4: POUR IN egg mixture. Mixture should set immediately at edges. Step 5: GENTLY PUSH cooked portions from edges toward the center with inverted turner so that uncooked eggs can reach the hot pan surface. Step 7: CONTINUE cooking, tilting pan and gently moving cooked portions as needed. When top surface of eggs is thickened and no visible liquid egg remains, PLACE filling on one side of the omelet. FOLD omelet in half with turner. With a quick flip of the wrist, turn pan and INVERT or SLIDE omelet onto plate. SERVE immediately.","2 EGGS,2 Tbsp. water, 1/8 tsp. salt , Dash pepper,1tsp  butter cup filling, such as shredded cheese, finely chopped ham, baby spinach 1/3 to 1/2")
# # Cheese cake
# insertRecipeData("cheesecake", "Step 1: Adjust the oven rack to the lower-middle position and preheat oven to 350°F (177°C). Step 2: Make the crust: If you’re starting out with full graham crackers, use a food processor or blender to grind them into fine crumbs. Pour into a medium bowl and stir in sugar until combined, and then stir in the melted butter; Mixture will be sandy; Try to smash/break up any large chunks. Pour into an ungreased 9-inch or 10-inch springform pan. With medium pressure using your hand, pat the crumbs down into the bottom and partly up the sides to make a compact crust. Do not pack down with heavy force because that makes the crust too hard. Simply pat down until the mixture is no longer crumby/crumbly and you can use the flat bottom of a small measuring cup to help smooth it all out if needed. Pre-bake for 10 minutes. Remove from the oven and place the hot pan on a large piece of aluminum foil. The foil will wrap around the pan for the water bath in step 4. Allow crust to slightly cool as you prepare the filling. Step 3: Make the filling: Using a handheld or stand mixer fitted with a paddle attachment, beat the cream cheese and granulated sugar together on medium-high speed in a large bowl until the mixture is smooth and creamy, about 2 minutes. Add the sour cream, vanilla extract, and lemon juice then beat until fully combined. On medium speed, add the eggs one at a time, beating after each addition until just blended. After the final egg is incorporated into the batter, stop mixing. To help prevent the cheesecake from deflating and cracking as it cools, avoid over-mixing the batter as best you can. You will have close to 6 cups of batter. Step 4: Prepare the simple water bath (see note): Watch my video tutorial below; the visual guide will assist you in this step. Boil a pot of water. You need 1 inch of water in your roasting pan for the water bath, so make sure you boil enough. I use an entire kettle of hot water. As the water is heating up, wrap the aluminum foil around the springform pan. Pour the cheesecake batter on top of the crust. Use a rubber spatula or spoon to smooth it into an even layer. Place the pan inside of a large roasting pan. Carefully pour the hot water inside of the pan and place in the oven. (Or you can place the roasting pan in the oven first, then pour the hot water in. Whichever is easier for you.) Step 5: Bake cheesecake for 55–70 minutes or until the center is almost set. If you notice the cheesecake browning too quickly on top, tent it with aluminum foil halfway through baking. When it’s done, the center of the cheesecake will slightly wobble if you gently shake the pan. Turn the oven off and open the oven door slightly. Let the cheesecake sit in the oven in the water bath as it cools down for 1 hour. Remove from the oven and water bath, then cool cheesecake completely uncovered at room temperature. Then cover and refrigerate the cheesecake for at least 4 hours or overnight. Step 6: Use a knife to loosen the chilled cheesecake from the rim of the springform pan, then remove the rim. Using a clean sharp knife, cut into slices for serving. For neat slices, wipe the knife clean and dip into warm water between each slice. Step 7: Serve cheesecake with desired toppings (see Note); Cover and store leftover cheesecake in the refrigerator for up to 5 days.","8-ounce blocks of full-fat cream cheese, 1 cup sugar, 1 cup Sour Cream, 1 teaspoon of vanilla extract, 2 teaspoon of lemon juice,  3 eggs")
# # Sushi
# insertRecipeData("sushi", "Step 1: Place the rice in a sieve; Rinse under cold running water, to remove any excess starch, until water runs clear. Place the rice and water in a large saucepan, covered, over high heat. Bring to the boil. Reduce heat to low and cook, covered, for 12 minutes or until all the water is absorbed. Remove from heat. Set aside, covered, for 10 minutes to cool slightly. Step 2: Combine the vinegar, sugar and salt in a small bowl. Transfer the rice to a large glass bowl. Use a wooden paddle to break up rice lumps while gradually adding the vinegar mixture, gently folding to combine. Continue folding and fanning the rice for 15 minutes or until rice is cool.Step 3: Place a sushi mat on a clean surface with slats running horizontally. Place a nori sheet, shiny-side down, on the mat. Use wet hands to spread one-sixth of the rice over the nori sheet, leaving a 3cm-wide border along the edge furthest away from you. Step 4: Place salmon and avocado along the centre of the rice. Hold filling in place while rolling the mat over to enclose rice and filling. Repeat with remaining nori, rice, salmon and avocado. Step 5 :Use a sharp knife to slice sushi widthways into 1.5cm-thick slices. Place on serving dishes with soy sauce, wasabi and pickled ginger, if desired.","2 1/2 cups (540g) koshihikari rice, 3 3/4 cups (935ml) cold water, 1/2 cup (125ml) rice vinegar, 2 tablespoons caster sugar, 1/2 teaspoon salt, 6 nori sheets, 200g fresh salmon, cut into 1cm-thick batons, 1 avocado, halved, stoned, peeled, thinly sliced, Light soy sauce, to serve, Wasabi paste, to serve, Pickled ginger, to serve")
# # Chicken Curry
# insertRecipeData("chicken_curry", "Step 1: Marinate the chicken in ginger garlic paste, lime juice and salt. This adds flavour from within and the acid helps tenderise the meat; step 2 Grind ginger, garlic, green chillies and onions to a paste with a little water. This becomes the base of the curry; Step by step picture collage showing how to make chicken curry in a pressure cooker; step 3 Pound spices roughly with a mortar pestle to break them down and release their flavour Saute them in hot oil to temper them; step 4 Brown the onion paste in oil that's tempered with whole spices. This is a slow cooking process where the onion paste is sauteed on a low flame till it changes colour and becomes a deep golden brown. You'll see a thin layer of oil floating on the top and the sides - thats when you know its cooked through and isn't raw anymore; step 5 Cook tomatoes and spices for a few minutes till the tomatoes become pulpy soft; step 6 Roast the chicken along with garam masala in the onion tomato mixture, tossing it for a minute or two till the gravy coats all the pieces; Pressure Cook for 15 minutes. If using a traditional pressure cooker, start on high heat and after the first whistle, reduce the heat to low. Wait till you hear two more whistles and then turn off the flame. Let the pressure release naturally. To make instant pot chicken curry, pressure cook on high for 8 minutes; step 7 Temper julienned ginger and slit green chillies in butter and ghee.This adds an extra layer of flavour and gives it a nice hint of ginger; step 8 Serve chicken curry with tempering poured on top. Sprinkle some chopped coriander for freshness and flavour;", "1 Chicken,2 Ginger Garlic Paste,3 Lime Juice ,4 Mustard oil ,5 Whole spices ,6 ginger&garlic , 7 onion ,8 green chillies ,9 Ground Spices,10 Tempering ( uses ghee, ginger and green chillies")
# # Ice cream
# insertRecipeData("ice_cream", "Step 1: Stir together condensed milk, cold milk, vanilla, and salt in a medium bowl. Set aside; step2 Beat heavy cream in a large bowl with an electric mixer until stiff peaks form. Fold milk mixture into whipped cream.; step 3 Pour into a shallow 2-quart dish, cover, and freeze for 4 hours, stirring once after 2 hours or when edges start to harden.; step 4 Serve or store in an airtight container for up to 10 days.;", "1 (14 ounce) can sweetened condensed milk,½ cup cold milk,1 tablespoon vanilla extract ,⅛ teaspoon salt,1 pint heavy cream")
# # Hot dog
# insertRecipeData("hot_dog", "Step 1: Gather the ingredients. Step 2: In a food processor, puree the onion, garlic, coriander, paprika, mustard seed, marjoram, and mace. Add the milk, egg white, sugar, salt, and pepper and combine well. Remove to a bowl and set aside. Step 3: One at a time, grind the pork, beef, and fat cubes through the fine blade of a meat grinder. Combine the 3 ingredients and grind them together. Step 4: In a large bowl, combine the pureed seasonings with the meat and mix with your hands. Wet your hands with cold water to prevent the mixture from sticking. Step 5: Refrigerate the mixture for 30 minutes and then grind it again. Prepare the casings. Step 6: Meanwhile, rinse the casing well under cool running water to remove the salt. Place it in a bowl of cool water and soak for 30 minutes. Step 7: After soaking, run cool water over the casing. Slip one end of the casing over the faucet nozzle and firmly hold it in place. Turn on the cold water, gently at first, and then more forcefully. This will flush out any salt in the casing and help you spot any breaks. Should you find one, simply snip out a small section of the casing. Step 8: Put the casing in a bowl of water and add a splash of white vinegar. (A tablespoon of vinegar per cup of water is sufficient.) The vinegar softens the casing and makes it more transparent, which in turn makes the hot dog look nicer. Leave the casing in the water and vinegar solution until you are ready to use it. Rinse it well and drain before stuffing. Step 9: Using a sausage stuffer, fill the casings with the meat mixture and twist them off into 6-inch links. Step 10: Parboil the links (but don't separate them) in simmering water for 20 minutes. Step 11: Place the ​franks in a bowl of ice water and chill. Step 12: Remove, pat dry, and refrigerate. You can refrigerate them for up to one week or freeze them for future use.","1/4 cup very finely minced onion, 1 small clove garlic, finely chopped, 1 teaspoon ground coriander, 1 teaspoon sweet paprika, 1/2 teaspoon ground mustard seed, 1/4 teaspoon dried marjoram, 1/4 teaspoon ground mace, 1/4 cup milk, 1 large egg white, 1 1/2 teaspoons sugar, 1 teaspoon salt, or to taste, 1 teaspoon freshly ground white pepper, 1 pound lean pork (cubed), 3/4 pound lean beef (cubed), 1/4 pound pork fat(cubed), 4 feet sheep casings (or small hog casings), about 1 1/2 inches in diameter, 1 tablespoon white vinegar")
# # Fried Rice
# insertRecipeData("fried_rice", "Step 1. A great vegetable fried rice recipe begins with well-cooked white rice. To do so, first, soak 1 cup of basmati rice (190 to 200 grams) in water for 30 mins; Then drain and set rice aside. Step 2. In a large pot add 4 to 4.5 cups of water, ½ teaspoon salt, and 2 to 3 drops of toasted sesame oil (or any cooking oil), and set over high heat on the stovetop. 3. Let the water reach a full boil. 4. Add the soaked and drained basmati rice. 5. Simmer the rice uncovered on medium to medium-high heat. 6. Cook rice until al dente, or just about cooked. It’s important that you do not overcook rice during this step in order to make the best Chinese fried rice dish. Overcooking rice or even cooking them completely, will break the rice grains while stir-frying. 7. Quickly strain rice in a colander, and let it cool completely at room temperature. Refrigerating rice for 30 minutes also helps. Simply cover cooked rice and refrigerate. 8. You can also rinse the rice with water so that the rice stops cooking. Use your hands to gently stir the cooked rice grains around while rinsing to ensure that every grain is cooled. Then drain thoroughly. 9. While rice is cooling, finely chop the veggies. 10. Now it’s time to stir-fry the vegetables. Start by heating 3 tablespoons of preferred cooking oil in a wok or a deep skillet. Add 1 whole star anise and fry for a few seconds, until the oil becomes fragrant. 11. Then add 1 teaspoon of finely chopped garlic, and ½ teaspoon finely chopped ginger. Cook for only a moment as you don’t want to brown or burn the garlic.12. Add the chopped spring onions, and sauté for a minute or two on medium-low to medium heat.13. Then add the finely chopped french beans – or, add these before the onions if you want a softer bean texture.14. Continue to stir-fry both the onions and french beans for 2 to 3 minutes over medium to medium-high heat.15. Now add the remaining finely chopped veggies, including mushrooms and celery. Increase the heat to high to thoroughly cook all of the ingredients.16. Continuously toss and stir while frying so that the veggies are uniformly cooked and do not get burnt – for about 4 to 6 minutes.17. Once the veggies have cooked, add 3 tablespoons of soy sauce. Stir well.18. Taste the dish, and then add ½ a teaspoon of ground black pepper and as much salt as you like.19. Mix well to combine the flavors.20. To the wok or skillet of veggies add the cooked and cooled rice – 1 cup or so at a time21. With the heat still on medium to medium-high, continue to gently mix and fry the vegetables with rice for a couple of minutes.22. Lastly, add more fresh chopped spring onions. You can either sprinkle on top as a garnish or mix into the fried rice to incorporate.", "")
# print(retrieveRecipeData())
# print(retrieveRecipeDataWithItemName("pizza"))
