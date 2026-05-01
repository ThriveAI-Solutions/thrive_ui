# Okta OIDC SSO Integration — Design

**Date:** 2026-05-01
**Branch:** `feat/okta-integration`
**Owner:** Rob Enderle
**Status:** Design — pending implementation plan
**Related deliverable:** ThriveAI 5/8 hand-off to HEALTHeLINK (HeL) — Okta integration metadata for HEALTHeCOMMUNITY (HeC) Portal.

## 1. Background and goal

HEALTHeINTELLIGENCE (this Streamlit app) currently authenticates users with a SHA-256 username/password against a SQLite `User` table, with sessions held in encrypted cookies via `EncryptedCookieManager`. Roles (`ADMIN`, `DOCTOR`, `NURSE`, `PATIENT` — `RoleTypeEnum`) drive page navigation, RAG vector-store filtering, and `VannaService` per-user caching.

The HEALTHeCOMMUNITY (HeC) Portal team wants to place a Portal badge that drops authenticated users directly into HEALTHeINTELLIGENCE via SSO. The Portal authenticates with Okta (Okta Classic) plus Duo MFA. The goal is to make HEALTHeINTELLIGENCE accept Okta OIDC logins from the Portal while keeping the codebase testable locally without depending on HeL's Okta tenant.

## 2. Goals and non-goals

**Goals**
- HEALTHeINTELLIGENCE accepts OIDC logins from an Okta IdP.
- Identity, role, and basic profile flow from Okta into the existing `User` row used by the rest of the app.
- The OIDC integration can be fully tested locally against a free Okta Developer org (no dependency on HeL's tenant for the 5/8 spike).
- The existing local username/password login remains available behind a config switch as a development and break-glass path.
- Produce the metadata HeL needs to register the app in Okta (the 5/8 deliverable).

**Non-goals (this round)**
- Full production AWS deployment with TLS cert, DNS, etc. — that's a HeL/AWS work item; this design just enumerates what we need from them.
- A mock OIDC server in CI. Tests run against a real Okta dev org for the spike. Pytest covers the synchronization logic only.
- Changing the existing `RoleTypeEnum`, RAG filtering logic, or `VannaService` cache behavior. The integration is designed to leave them untouched.
- RP-initiated logout (logging the user out of Okta itself). User stays signed in to the Portal.

## 3. Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Two auth modes selected by config**: `oidc` or `local`. Only one is active at a time. | Keeps local login as a dev/break-glass path while making prod a single, hardened code path. |
| 2 | **JIT user provisioning** with **Okta group claim → role mapping**. Default role on no-match is `DOCTOR`. | Matches "give everyone more lax permissions by default" guidance from stakeholders. New Portal users "just work" without manual admin setup. |
| 3 | **Identity link**: `okta_sub` is canonical; `email` is the bootstrap key on first login. | Stable across email changes; allows pre-provisioning by email if admins want it later. |
| 4 | **Implementation**: Streamlit's native `st.login()` / `st.user` (added in 1.42). | Streamlit 1.43.2 is already in `pyproject.toml`. Uses Authlib under the hood. Smallest custom auth code surface. Matches the Streamlit doc HeL is likely to read. |
| 5 | **Testing**: Free Okta Developer org for manual end-to-end testing during the spike. No mock OIDC server. | Closest to production behavior. Covers the actual demo case for HeL. CI tests cover the sync logic only. |
| 6 | **Logout** clears the local Streamlit session and redirects to a configurable URL (default: HeC Portal). No RP-initiated Okta logout. | User entered HEALTHeINTELLIGENCE *via* the Portal badge; logout should return them to the Portal, not nuke their Okta session everywhere. |

## 4. Architecture

### 4.1 Module layout and dispatcher

A new helper module `utils/okta_auth.py` owns all OIDC-side concerns. `utils/auth.py` becomes a thin dispatcher selected by `secrets.toml`:

```
app.py
  └─ check_authenticate()                       [utils/auth.py]
       │
       ├─ if [auth] block present in secrets:   → handle_oidc_auth()    [utils/okta_auth.py]
       │       └─ uses st.login() / st.user
       │       └─ on first authenticated render: sync_okta_user_to_db()
       │       └─ populates session state to mirror local-login shape
       │
       └─ else:                                 → existing cookie-based path
                (verify_user_credentials, EncryptedCookieManager, show_login)
```

### 4.2 Critical invariant

After `check_authenticate()` returns, both paths leave the app in **the same session-state shape**. Today that shape is:

- `st.session_state.cookies["user_id"]` — JSON-encoded integer User PK
- `st.session_state.cookies["role_name"]` — string role name
- `st.session_state.user_role` — integer `RoleTypeEnum.value`
- `st.session_state.username` — display name (e.g., `"Rob Enderle"`)
- All preference flags populated by `set_user_preferences_in_session_state()`

The OIDC path mirrors this shape so that nothing downstream (`VannaService`, RAG filters, message persistence, page-level admin gates) needs to know which path ran. Specifically: in OIDC mode the actual *authentication* state lives in Streamlit's `st.user` (managed by `st.login()`), but we still write `st.session_state.cookies["user_id"]` and `cookies["role_name"]` so that downstream readers like `_get_current_user_id()` (`orm/functions.py`) and `extract_user_context_from_streamlit()` keep working unmodified. The cookie values become read-only mirrors, not the source of truth, in OIDC mode.

### 4.3 What does *not* change

- `orm/models.py` `RoleTypeEnum` values (0/1/2/3 ordering preserved).
- `set_user_preferences_in_session_state` and `save_user_settings`.
- `VannaService.invalidate_cache_for_user`, `extract_user_context_from_streamlit`.
- RAG role filtering in `utils/chromadb_vector.py` / `utils/milvus_vector.py`.
- The existing logout button placement and copy in the sidebar.
- `Message` persistence and `_get_current_user_id()` semantics.

## 5. Schema changes

Two new nullable columns on `thrive_user`:

| Column | Type | Index | Purpose |
|---|---|---|---|
| `okta_sub` | VARCHAR(255) | UNIQUE (nullable) | Canonical Okta identifier. NULL for local-only users. |
| `email` | VARCHAR(320) `COLLATE NOCASE` | UNIQUE (nullable) | Bootstrap match on first OIDC login; useful for future pre-provisioning. SQLite `COLLATE NOCASE` makes the unique index case-insensitive natively. Matching code lowercases the input before lookup as a belt-and-suspenders measure. |

`username` and `password` columns are retained — still used by local mode. New columns are nullable so existing seeded users (`thriveai-kr`, etc.) keep working in local mode without any backfill required.

The implementation plan will check whether the project uses Alembic migrations or `Base.metadata.create_all()`. Based on a glance at the seeding pattern in `orm/models.py`, it appears to be the latter, in which case a hand-written ALTER TABLE script under `scripts/` is appropriate for the SQLite dev database. We will not change the migration pattern as part of this work.

## 6. OIDC login flow

```
1. User visits the app → check_authenticate() runs.
2. If not st.user.is_logged_in:
     show "Sign in with HEALTHeCOMMUNITY (Okta)" button → calls st.login("okta")
     st.stop()
3. Streamlit redirects to the configured Okta authorization endpoint with PKCE.
4. User authenticates at Okta (Duo MFA enforced on the Okta side, transparent to us).
5. Okta redirects back to /oauth2callback. Streamlit validates the ID token and populates st.user.
6. Page reruns → check_authenticate() now sees st.user.is_logged_in == True.
7. sync_okta_user_to_db() runs:
     - Look up User WHERE okta_sub == st.user.sub          → if found, use it.
     - Else look up User WHERE lower(email) == lower(st.user.email)
                                                            → if found, stamp okta_sub onto it.
     - Else JIT-create:
         User(
           username   = st.user.email,           # admin can rename later
           password   = NULL,                    # local login disabled for this user
           email      = st.user.email,
           okta_sub   = st.user.sub,
           first_name = st.user.given_name or "",
           last_name  = st.user.family_name or "",
           user_role_id = role_id_from_groups(st.user.groups),  # default DOCTOR if no match
           # all preference defaults from create_user()
         )
     - Update user.user_role_id from the group claim on every login (Okta is source of truth
       for OIDC users — any in-app admin role change to a user with non-NULL okta_sub will be
       overwritten on their next login. Users with NULL okta_sub, i.e. local-mode users, are
       unaffected by this rule).
8. Populate session state to match local-login shape (see §4.2).
9. log_login(user_id=user.id, username=user.username, success=True)  # reuse existing logging.
10. Continue to pg.run() as normal.
```

## 7. Group-claim → role mapping

A small mapping table in `utils/okta_auth.py`:

```python
OKTA_GROUP_TO_ROLE = {
    "thriveai-admin":   RoleTypeEnum.ADMIN,    # 0
    "thriveai-doctor":  RoleTypeEnum.DOCTOR,   # 1
    "thriveai-nurse":   RoleTypeEnum.NURSE,    # 2
    "thriveai-patient": RoleTypeEnum.PATIENT,  # 3
}
DEFAULT_ROLE_IF_NO_GROUP_MATCH = RoleTypeEnum.DOCTOR
```

**Resolution rule:** read `st.user.groups` (a JSON array of strings; claim name configurable, default `groups`). Of the groups that appear in `OKTA_GROUP_TO_ROLE`, pick the one whose role has the **lowest enum value** (highest privilege). If no groups match, use `DEFAULT_ROLE_IF_NO_GROUP_MATCH = DOCTOR`.

The HeL ↔ ThriveAI contract for the claim is: a JSON array of strings under the agreed claim name. If HeL prefers a different claim name or membership format, only the constants above change.

## 8. Logout flow

When the user clicks **Log Out** in the sidebar, the dispatcher in `utils/auth.py` chooses the right logout path based on `auth.mode`.

**OIDC-mode logout:**

1. `VannaService.invalidate_cache_for_user(user_id, user_role)`
2. Clear local session state: `st.session_state.messages = []`, `st.session_state.selected_llm_provider = None`, `st.session_state.selected_llm_model = None`, `st.session_state._vn_instance = None`. Also clear the mirrored cookie values (`cookies["user_id"]`, `cookies["role_name"]`) per §4.2.
3. Call `st.logout()` to clear Streamlit's auth cookie.
4. Redirect to `auth.post_logout_redirect_url` (the HeC Portal URL by default).

We do **not** issue an OIDC RP-initiated logout (no call to Okta's `/v1/logout`). The user remains signed in to Okta and the Portal, which is the desired behavior for a Portal-launched session.

**Local-mode logout:** unchanged — same `EncryptedCookieManager`-based flow currently in `utils/auth.py:36–61`. No edit required.

## 9. Configuration

### 9.1 `secrets.toml` additions (OIDC mode)

```toml
[auth]
mode = "oidc"                                                          # or "local" for dev/break-glass
redirect_uri = "https://app.healtheintelligence.example.com/oauth2callback"
cookie_secret = "<32+ random bytes — different from [cookie].password>"
post_logout_redirect_url = "https://portal.healthecommunity.org/"

[auth.okta]
client_id = "<from Okta app>"
client_secret = "<from Okta app>"
server_metadata_url = "https://<okta-tenant>.okta.com/.well-known/openid-configuration"
client_kwargs = { scope = "openid email profile groups" }
```

If `[auth]` is absent or `mode = "local"`, the dispatcher falls through to the existing cookie/password flow. No change to the `[cookie]` section — `EncryptedCookieManager` is still used by local mode, and incidentally for non-auth cookie usage if any creeps in.

### 9.2 Local-mode config (unchanged)

```toml
[cookie]
password = "<random-strong-secret>"
```

### 9.3 Switching between modes

To run locally against Okta dev: set `[auth]` block populated, `mode = "oidc"`.
To run locally with the existing username/password form: comment out the `[auth]` block or set `mode = "local"`.

No code change required to switch.

## 10. Deliverable for HeL (5/8)

This table is the contract we hand to HeL so they can configure the OIDC app in their Okta Classic tenant:

| Field | Value |
|---|---|
| Application type | OIDC — Web Application |
| Grant type | Authorization Code (with PKCE) |
| Sign-in redirect URI | `https://<our-prod-host>/oauth2callback` (and `http://localhost:8501/oauth2callback` for ThriveAI dev) |
| Sign-out redirect URI | `https://<our-prod-host>/` (or HeC Portal URL) |
| Required scopes | `openid`, `email`, `profile`, `groups` |
| Required claims in ID token | `sub`, `email`, `email_verified`, `given_name`, `family_name`, `groups` (custom) |
| Group claim format | JSON array of strings, claim name `groups` |
| Group names HeL must create and assign | `thriveai-admin`, `thriveai-doctor`, `thriveai-nurse`, `thriveai-patient` |
| Token endpoint auth method | `client_secret_basic` |
| Logout behavior | App-side session clear only; no RP-initiated Okta logout (user stays signed into Portal) |
| MFA | Enforced at Okta/Duo level; transparent to app |
| Open prereq from HeL/AWS side | Production hostname + TLS cert (current `IP:port` access cannot serve as an OIDC redirect URI in prod) |

The doc handed to HeL will wrap this table with one page of prose describing the high-level integration shape and the JIT/role-mapping rule so HeL knows what claims they must emit.

## 11. Local testing — Okta Developer org walkthrough

Steps to set up a free Okta Developer org for end-to-end local testing:

1. **Sign up** at `https://developer.okta.com/signup/` (free, no card). You receive a tenant URL like `https://dev-12345678.okta.com`.
2. **Create the OIDC app** via the Application Integration Wizard: Applications → Create App Integration → OIDC → Web Application.
   - Sign-in redirect URI: `http://localhost:8501/oauth2callback`
   - Sign-out redirect URI: `http://localhost:8501/`
   - Save the `Client ID` and `Client secret`.
3. **Create a custom `groups` claim**: Security → API → Authorization Servers → `default` → Claims → Add Claim.
   - Name: `groups`
   - Include in: ID token (Always)
   - Value type: Groups
   - Filter: `Matches regex .*` (or `Starts with thriveai-` if you only want app-relevant groups in the token).
4. **Create groups**: Directory → Groups → Add Group, four times: `thriveai-admin`, `thriveai-doctor`, `thriveai-nurse`, `thriveai-patient`.
5. **Create test users**: Directory → People → Add Person. Create at least one per role, e.g. `admin@test.local`, `doctor@test.local`, `nurse@test.local`, `patient@test.local`. Activate immediately and skip the activation email; set a password manually.
6. **Group memberships**: assign each test user to the corresponding group.
7. **App assignments**: Applications → your app → Assignments → assign all four groups (or all four users individually).
8. **Wire up `secrets.toml`** locally:
   ```toml
   [auth]
   mode = "oidc"
   redirect_uri = "http://localhost:8501/oauth2callback"
   cookie_secret = "<32+ random bytes>"
   post_logout_redirect_url = "http://localhost:8501/"

   [auth.okta]
   client_id = "<from step 2>"
   client_secret = "<from step 2>"
   server_metadata_url = "https://dev-12345678.okta.com/oauth2/default/.well-known/openid-configuration"
   client_kwargs = { scope = "openid email profile groups" }
   ```
9. **Run** `uv run streamlit run app.py`, click "Sign in with HEALTHeCOMMUNITY (Okta)", log in as each test user, and verify:
   - Each user's role lands correctly in the app (admin user sees Admin Analytics page; doctor user does not).
   - On first login per user, a row is JIT-created in `thrive_user` with the correct `okta_sub`, `email`, and `user_role_id`.
   - On subsequent logins, the same row is reused (no duplicates).
   - Logout returns the user to the configured `post_logout_redirect_url`.

Initial setup is roughly 30 minutes. Switching back to local mode is a one-line edit to `secrets.toml`.

## 12. Tests

Unit / integration tests in `tests/` covering the synchronization logic. No real OIDC traffic; the `st.user`-shaped object is faked.

| Test | What it verifies |
|---|---|
| `test_okta_auth.py::test_jit_creates_new_user` | Given a fake `st.user` for a brand-new identity, `sync_okta_user_to_db` creates the row with default DOCTOR role. |
| `test_okta_auth.py::test_existing_user_matched_by_sub` | When `okta_sub` already exists on a row, that row is used (not a new one). |
| `test_okta_auth.py::test_existing_user_matched_by_email_stamps_sub` | Bootstrap path: row with matching email but no `okta_sub` gets the sub stamped. |
| `test_okta_auth.py::test_group_claim_admin_wins` | When user is in both `thriveai-admin` and `thriveai-doctor`, ADMIN role is selected. |
| `test_okta_auth.py::test_no_group_match_defaults_to_doctor` | When no groups in the claim match, the default role is DOCTOR. |
| `test_okta_auth.py::test_role_updates_on_subsequent_login` | If groups change between logins, the role is updated to match. |
| `test_auth_dispatcher.py::test_local_mode_unchanged` | When `[auth]` is absent from secrets, the existing flow runs and is functionally unchanged. |

The full OIDC flow is validated **manually** against the Okta Developer org per §11.

## 13. Risks and open items

- **Okta Classic vs. Identity Engine UI parity.** HeL runs Okta Classic. The Okta Developer org may run on the Identity Engine. The OIDC protocol surface our app touches is identical, so the application code is portable, but the *UI steps HeL follows* differ from the steps in §11. The deliverable should call this out so HeL doesn't try to copy ThriveAI's screenshots verbatim.
- **Production hostname and TLS cert.** Okta will reject non-HTTPS redirect URIs in production. The current `IP:port` access path cannot serve as the OIDC redirect target. HeL or the AWS team must provide a DNS name and TLS cert before go-live. Local dev (`http://localhost:8501`) is exempt because Okta allows `http://localhost` redirect URIs for development.
- **Existing seeded users.** The existing `thriveai-*` users have no `okta_sub` or `email`. When mode flips to `oidc`, those rows continue to exist but will not be matched by an OIDC login. If any of them must be preserved (with their `Message` history), pre-fill the `email` column for each row before the first OIDC login. This is a one-line script. The implementation plan will include it.
- **Duo MFA.** Transparent to the app; Duo enforcement happens during the Okta auth code flow. No app changes required.
- **`cookie_secret` rotation.** `st.login()` requires a `cookie_secret`. Rotating it logs out all existing OIDC sessions. Document this in operations notes; it's not a blocker.
- **Refresh tokens.** `st.user` lifetime is governed by Streamlit's auth cookie, not the OIDC `refresh_token`. The app does not store or refresh tokens directly. This is acceptable for the spike. If long-lived sessions become required, revisit.

## 14. Out of scope (revisit later)

- Edge-terminated auth via AWS ALB OIDC or `oauth2-proxy`. Considered and rejected for this round (see brainstorm transcript). May revisit after go-live.
- A mock OIDC server in CI. Not built now. Add only if the synchronization logic grows complex enough to warrant integration tests over real OIDC traffic.
- SCIM provisioning / Okta-driven user lifecycle (deactivation, role removal in real time). JIT covers create and per-login role refresh; deactivation in Okta will simply prevent future logins, which is sufficient for the spike.
- Separate dev / staging Okta tenants for HeL. The free Developer org is sufficient for ThriveAI-side validation; HeL uses their own tenant for their environments.

## 15. Implementation sketch (file-level changes)

The implementation plan (next document) will detail this. High-level files affected:

- **New** `utils/okta_auth.py` — `handle_oidc_auth`, `sync_okta_user_to_db`, `role_id_from_groups`, `is_oidc_mode`.
- **Modified** `utils/auth.py` — dispatcher; existing `check_authenticate` body becomes `_handle_local_auth`.
- **Modified** `orm/models.py` — add `okta_sub` and `email` columns to `User`.
- **New** `scripts/migrate_add_okta_columns.py` — ALTER TABLE for SQLite dev DB; only needed if not using Alembic.
- **Modified** `pyproject.toml` — add `Authlib` dep if not transitively present (Streamlit's auth pulls it).
- **New** `tests/test_okta_auth.py`, `tests/test_auth_dispatcher.py` — unit tests per §12.
- **New** `docs/superpowers/specs/2026-05-08-okta-integration-handoff-to-hel.md` — the customer-facing deliverable derived from §10 (when ready to send).

Estimated effort: 1–2 focused days for code + tests; ~30 minutes for the Okta dev org setup; another 1–2 hours to write the HeL hand-off doc.
