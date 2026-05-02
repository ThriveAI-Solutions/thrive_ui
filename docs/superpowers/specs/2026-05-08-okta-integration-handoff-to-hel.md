# HEALTHeINTELLIGENCE — Okta OIDC Integration Requirements

**To:** HEALTHeLINK (HeL) — Robert Irvine, Ryan, Casey, Alyssa
**From:** ThriveAI — Rob Enderle
**Date:** 2026-05-08
**Status:** Requirements for HeL to configure the HEALTHeINTELLIGENCE app integration in HeC Portal Okta (Classic).

## 1. Summary

HEALTHeINTELLIGENCE integrates with the HEALTHeCOMMUNITY (HeC) Portal as a standard OIDC Relying Party. The app uses Streamlit's native OIDC support (Authlib under the hood). When a user clicks the Portal badge for HEALTHeINTELLIGENCE, they are redirected to the Okta authorization endpoint, authenticate (with Duo MFA on the Portal/Okta side), and are returned to the app with an ID token. The app reads identity, basic profile, and group membership from the ID token to drive role-based access.

The integration is **app-side**. No edge proxy or AWS ALB OIDC integration is required. The app handles the OIDC flow internally via `st.login()`.

## 2. What HeL must configure in Okta

| Field | Value |
|---|---|
| Application type | OIDC — Web Application |
| Grant type | Authorization Code (with PKCE) |
| Sign-in redirect URI | `https://<our-prod-host>/oauth2callback` (final hostname TBD by AWS team) and `http://localhost:8501/oauth2callback` (ThriveAI dev) |
| Sign-out redirect URI | `https://<our-prod-host>/` (or HeC Portal landing URL) |
| Required scopes | `openid`, `email`, `profile`, `groups` |
| Required claims in ID token | `sub`, `email`, `email_verified`, `given_name`, `family_name`, `groups` (custom) |
| Group claim format | JSON array of strings under claim name `groups` |
| Group names HeL must create and assign | `thriveai-admin`, `thriveai-doctor`, `thriveai-nurse`, `thriveai-patient` |
| Token endpoint authentication method | `client_secret_basic` |
| Logout behavior | App-side session clear only. We do NOT issue OIDC RP-initiated logout. User remains signed in to Okta and the Portal. |
| MFA | Enforced at Okta/Duo level. Transparent to the app. |
| Open prerequisite (HeL/AWS) | Production hostname + TLS cert. Current `IP:port` access cannot serve as an OIDC redirect target in production. |

## 3. What HeL must give back to ThriveAI

For each environment (dev/stage/prod, or however HeL splits them):

- Issuer URL — the `https://<okta>/oauth2/default` form, or whichever authorization server you wire up.
- `Client ID`
- `Client secret`
- Confirmation that the four `thriveai-*` groups exist and are emitted in the `groups` claim of the ID token.
- The exact production hostname HeL plans to use for HEALTHeINTELLIGENCE (so we register the right redirect URIs).

## 4. Authorization model on the app side

ThriveAI maps the `groups` claim to internal roles as follows:

| Okta group | App role | Capabilities |
|---|---|---|
| `thriveai-admin` | `Admin` | All pages, training data management, analytics, feedback dashboard. |
| `thriveai-doctor` | `Doctor` | Chat, user settings, full data view. |
| `thriveai-nurse` | `Nurse` | Chat, user settings, restricted data view. |
| `thriveai-patient` | `Patient` | Chat, user settings, most restricted data view. |

If a user is in multiple `thriveai-*` groups, the highest-privilege one wins. If a user is in **none** of these groups, they are treated as a `Doctor` by default (per stakeholder guidance to be permissive on go-live).

A user who authenticates successfully but is in no `thriveai-*` group still gets in. If you want to gate access entirely on group membership, tell us and we will change the default to "deny."

## 5. User provisioning

HEALTHeINTELLIGENCE uses **Just-in-Time (JIT) provisioning**. There is no SCIM connector. The first time a user authenticates via Okta, a row is created in our internal user table keyed off the `sub` claim, with email and name copied from the ID token. The user's role is refreshed from the `groups` claim on **every** login, so role changes in Okta take effect on the user's next login (no manual sync needed).

There is no "deactivate" path on the app side beyond Okta deactivation — if you remove a user in Okta, they cannot log in. Their data row in our app remains for audit purposes; we can purge on request.

## 6. Note on Okta Classic vs Identity Engine

ThriveAI's local validation environment is a free Okta Developer org. That org runs the Identity Engine. HeC Portal runs Okta Classic. The OIDC protocol surface our app touches is identical between the two engines, so no app changes are required. The **Okta admin UI workflows differ**, however — the steps to add a custom claim or configure an app are similar in spirit but laid out differently. Use HeL's existing Okta Classic playbooks; do not copy our screenshots.

## 7. Open questions to confirm by go-live

1. Is `groups` an acceptable claim name, or does HeL standardize on a different name (`group_membership`, `roles`, etc.)? If different, tell us and we will update one constant.
2. Will HeL provide a single tenant for all environments or separate tenants per environment? We can support either.
3. Should logout return the user to a specific Portal URL? Default in our config is the value HeL gives us; otherwise we'll redirect to HeC Portal's landing page.
4. Are there additional claims HeL wants us to enforce or display (e.g., `department`, `npi`, `organization_id`)? If so, list them and the field type, and we'll add them to the app.

## 8. Reference

The ThriveAI-side design and implementation plan are in:
- `docs/superpowers/specs/2026-05-01-okta-oidc-integration-design.md`
- `docs/superpowers/plans/2026-05-02-okta-oidc-integration.md`
