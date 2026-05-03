# Admin Guide — Video-QA

## Getting Admin Access

Admin accounts are set directly in the database. To promote a user:

```python
# Run from the project root
import sys; sys.path.insert(0, ".")
from api import db
db.set_user_role("<user_id>", "admin")
```

Or via the admin panel itself: an existing admin can promote other users.

Once your account has the `admin` role, a link to the **Admin Panel** (`/admin`) appears in the navigation bar after login.

---

## Admin Panel Features

### System Stats (`/admin/stats`)

| Metric | Description |
|---|---|
| Total users | All registered accounts |
| Admin count | Accounts with `role = admin` |
| Total videos | All videos across all users |
| Ready videos | Successfully processed and indexed |
| Failed videos | Processing failed (error stored) |
| Total chunks | Sum of indexed transcript chunks |

### User Management (`/admin/users`)

The user list shows every account with:
- Email, role, auth provider (local / google)
- Email verification status
- Video count and ready-video count
- Account creation time

#### Promote / demote a user

Click the role badge next to any user to toggle between `user` and `admin`.

**You cannot demote your own account.** This prevents accidental lockout.

#### API endpoint

```
POST /admin/users/{user_id}/role
Authorization: Bearer <admin_token>
Content-Type: application/json

{ "role": "user" }   // or "admin"
```

Returns `{ "user_id": "...", "role": "user" }`.

### Video Management (`/admin/videos`)

View all videos across all users, including:
- Owner email and user ID
- Filename, status, error message (if failed)
- Chunk count, creation and update timestamps

Admins can delete any video via `DELETE /videos/{video_id}` using an admin token.

---

## Rate Limits

| Endpoint | Limit | Window |
|---|---|---|
| `POST /auth/register` | 5 attempts | per hour per IP |
| `POST /auth/login` | 10 attempts | per 15 minutes per IP |
| `POST /auth/request_reset` | 3 OTPs | per hour per IP |
| `POST /auth/resend_verification` | 5 sends | per hour per IP |
| `POST /process_url` | 3 URLs | per hour per user |

All rate-limit events are stored in the `rate_limit_events` table in SQLite. To clear all limits (e.g. during development):

```python
from api import db
with db._conn() as c:
    c.executescript("DELETE FROM rate_limit_events")
```

---

## Email Configuration

The system sends transactional emails for:
- Email verification (on register and via resend)
- Password reset OTP

Configure SMTP by setting two environment secrets:

| Secret | Value |
|---|---|
| `SMTP_EMAIL` | The Gmail address to send from |
| `SMTP_PASSWORD` | A Gmail App Password (Settings → Security → 2-Step Verification → App Passwords) |

The system uses `smtp.gmail.com:587` with STARTTLS. If either secret is missing, email delivery is silently skipped and the OTP/code is only logged server-side.

To test connectivity:
```python
from api.email import _send_sync
_send_sync("you@example.com", "Test", "Body")
```

Returns `True` on success, `False` on failure (never raises).

---

## Security Notes

- **JWT secret** — Set `JWT_SECRET` to a strong random string (≥ 32 characters). Changing it invalidates all existing sessions.
- **Password reset tokens** — OTPs expire in 10 minutes and are consumed on first use.
- **Email enumeration protection** — `/auth/request_reset` always returns the same generic message regardless of whether the email exists.
- **Token invalidation** — Changing a password invalidates all tokens issued before the change.
- **Admin RBAC** — Every admin endpoint calls `require_admin`, which re-reads the role from the database on every request — role changes take effect immediately.

---

## Logs

The server logs at `INFO` level to stdout. Key log prefixes:

| Prefix | Meaning |
|---|---|
| `[otp]` | OTP generated for password reset |
| `[email_verify]` | Verification code generated |
| `[email] ✓ Sent` | Email delivered successfully |
| `[email] ✗` | Email delivery failed (SMTP error detail follows) |
| `[admin]` | Admin role change |
| `[worker]` | Background video processing stage |
| `[ask]` | Question-answering warnings |

Pipeline-internal logs (retrieval, reranking, embedding) are suppressed at `WARNING` level in production to keep logs clean.

---

## Database

SQLite database is at `data/video_qa.db`. Key tables:

| Table | Contents |
|---|---|
| `users` | id, email, password_hash, role, email_verified, otp_hash, otp_expiry |
| `videos` | video_id, user_id, filename, status, progress, stage, chunk_count, error |
| `jobs` | job_id, video_id, status, progress, stage, error |
| `rate_limit_events` | ip, event_type, timestamp |

For production deployments, set `DATABASE_URL` to a PostgreSQL connection string — the schema is created automatically on startup.

---

## Deployment Checklist

- [ ] `JWT_SECRET` set to a strong random value
- [ ] `SMTP_EMAIL` and `SMTP_PASSWORD` configured (or email delivery disabled by omitting them)
- [ ] `ffmpeg` installed on the system
- [ ] `data/` directory writable
- [ ] Storage backend confirmed: `GET /health` returns `"db_backend": "sqlite"` or `"postgresql"`
- [ ] At least one admin account created
- [ ] `GET /health` returns `"status": "ok"`
