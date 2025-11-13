This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

# Phantom Air/Hotels

Design Discovery Documentation: https://github.com/velocitatem/PHANTOM/wiki/Design-Discovery

> This webapp serves two modes `{HOTEL,AIRLINE}` which are given by an env variable

The webapp should serve under the / route the landing page which for both platforms is very similar. We define a set of components like Hero, Card, Button, Link ... This we can then pass to specific components each mode might demand that makes it behave differently, hotel cards showing hotel rooms from database and airline cards showing flights from database and each fetching prices from the pricing provider with a different HTTP parameter.

- globally we define a middleware.ts which is our switcher for modes.
- /app will have (airline) and (hotel) children which each have a layout.tsx and page.tsx where /app also has a parent layout defining layout.tsx and globals.css for any shared styling to avoid repretition.
- /components/ is gonna have ui/ which defines things like Button, Card, DatePicker with generic definitions and any tracking or observation code. We then define feats/airline/ and feats/hotel/ as children of components with specific components like AirlineHero and HotelCard.
- in /styles/ we define airline.css and hotel.css to tailor accents and styling for each.

## How to Run

```sh
# install deps
npm install

# set store mode (hotel or airline)
export STORE_MODE=hotel

# run dev server
npm run dev
```

Server runs on `http://localhost:3000`

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `HOSTNAME` | Server hostname | `localhost` | `localhost` |
| `STORE_MODE` | Mode switch for platform | `hotel` | `hotel` or `airline` |
| `NEXT_PUBLIC_API_BASE` | Public API base URL | `http://localhost:3000` | `http://localhost:3000` |
| `NEXT_PUBLIC_APP_ENV` | Application environment | `dev` | `dev`, `prod` |
| `NEXT_PUBLIC_HOVER_THRESHOLD` | Hover dwell threshold (ms) | `1200` | `1200` |
| `BACKEND_URL` | Backend service URL | `http://localhost:5000` | `http://localhost:5000` |

## Routes

### Public Pages
- `/` — Landing page (mode-aware root)
- `/hotel` — Hotel mode landing
- `/hotel/products` — Hotel catalog
- `/airline` — Airline mode landing
- `/airline/products` — Flight catalog
- `/admin/experiments` — Experiment management UI

### API Routes
- `GET /api/session` — Fetch or create session, sets httpOnly cookie
- `GET /api/pricing?productId=X&sessionId=Y&experimentId=Z` — Get product price from provider
- `POST /api/ingest` — Ingest event to Kafka via backend
- `GET /api/admin/experiments` — List all experiments
- `POST /api/admin/experiments/start` — Start new experiment for session
- `POST /api/admin/experiments/stop` — Stop experiment by ID

## Event Catalog

All events are ingested via `POST /api/ingest` and follow the `EventBase` schema. Below are the 17 canonical events:

| Event Name | Category | Payload Example |
|------------|----------|-----------------|
| `session_start` | Session | `{ sessionId, experimentId?, storeMode, ts, page, eventName, userAgent? }` |
| `page_view` | Navigation | `{ sessionId, experimentId?, storeMode, ts, page: "/hotel", eventName: "page_view" }` |
| `view_item_page` | Discovery | `{ sessionId, storeMode, ts, page: "/hotel/products", productId: "H001", eventName: "view_item_page" }` |
| `learn_more_about_item` | Discovery | `{ sessionId, storeMode, ts, page, productId, eventName: "learn_more_about_item" }` |
| `add_item_to_cart` | Cart | `{ sessionId, storeMode, ts, page, productId, eventName: "add_item_to_cart" }` |
| `remove_item` | Cart | `{ sessionId, storeMode, ts, page, productId, eventName: "remove_item" }` |
| `checkout_start` | Cart | `{ sessionId, storeMode, ts, page, eventName: "checkout_start" }` |
| `purchase_complete` | Cart | `{ sessionId, storeMode, ts, page, eventName: "purchase_complete", metadata?: { total: 500 } }` |
| `search` | Filter/Search | `{ sessionId, storeMode, ts, page, eventName: "search", metadata: { query: "paris" } }` |
| `filter_for_date` | Filter/Search | `{ sessionId, storeMode, ts, page, eventName: "filter_for_date", metadata: { from: "2025-01-15", to: "2025-01-20" } }` |
| `filter_for_amenities` | Filter/Search | `{ sessionId, storeMode, ts, page, eventName: "filter_for_amenities", metadata: { amenities: ["wifi", "pool"] } }` |
| `filter_for_price` | Filter/Search | `{ sessionId, storeMode, ts, page, eventName: "filter_for_price", metadata: { min: 100, max: 500 } }` |
| `sort_change` | Filter/Search | `{ sessionId, storeMode, ts, page, eventName: "sort_change", metadata: { sort: "price_asc" } }` |
| `hover_over_title` | Dwell signal | `{ sessionId, storeMode, ts, page, productId?, eventName: "hover_over_title", metadata: { duration: 1500 } }` |
| `hover_over_paragraph` | Dwell signal | `{ sessionId, storeMode, ts, page, productId?, eventName: "hover_over_paragraph", metadata: { duration: 2000 } }` |
| `hover_over_link` | Dwell signal | `{ sessionId, storeMode, ts, page, productId?, eventName: "hover_over_link", metadata: { href: "/hotel/products" } }` |
| `hover_over_button` | Dwell signal | `{ sessionId, storeMode, ts, page, productId?, eventName: "hover_over_button", metadata: { label: "Book Now" } }` |

## Architecture

### Route Groups
- `(hotel)` — Hotel mode pages
- `(airline)` — Airline mode pages
- `api/*` — API routes (session, pricing, ingest, admin)

### Middleware Flow
1. Request arrives at Next.js
2. Session middleware checks for `phantom_session_id` cookie
3. If missing, `/api/session` mints new session + sets cookie
4. Store mode (`STORE_MODE` env) determines rendered page variant
5. Client-side components fetch pricing via `/api/pricing`
6. User interactions emit events to `/api/ingest` → Kafka
