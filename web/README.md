This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# Phantom Air/Hotels

Design Discovery Documentation: https://github.com/velocitatem/PHANTOM/wiki/Design-Discovery

> This webapp serves two modes `{HOTEL,AIRLINE}` which are given by an env variable

The webapp should serve under the / route the landing page which for both platforms is very similar. We define a set of components like Hero, Card, Button, Link ... This we can then pass to specific components each mode might demand that makes it behave differently, hotel cards showing hotel rooms from database and airline cards showing flights from database and each fetching prices from the pricing provider with a different HTTP parameter.

- globally we define a middleware.ts which is our switcher for modes.
- /app will have (airline) and (hotel) children which each have a layout.tsx and page.tsx where /app also has a parent layout defining layout.tsx and globals.css for any shared styling to avoid repretition.
- /components/ is gonna have ui/ which defines things like Button, Card, DatePicker with generic definitions and any tracking or observation code. We then define feats/airline/ and feats/hotel/ as children of components with specific components like AirlineHero and HotelCard.
- in /styles/ we define airline.css and hotel.css to tailor accents and styling for each.
