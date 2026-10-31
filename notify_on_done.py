#!/usr/bin/env python3
"""Watch for the Table 1 pipeline to finish, then deliver the completed table.

Runs independently of any Claude session, because the pipeline takes hours and there
may be nobody and nothing attached when it lands. Delivery degrades gracefully:

  1. Email, if SMTP credentials exist at ~/.config/btp_mail.json. That file is read
     only by this script and is never printed or logged. Create it yourself:

         {"user": "you@gmail.com", "app_password": "....", "to": "you@wherever"}

     Use a Google *app password*, not your account password, and chmod 600 it.
     If the file is absent, this step is skipped silently -- no credential is
     ever requested interactively.
  2. A desktop notification, if a session is logged in.
  3. Always: the table written to table1_FINAL.txt next to this script.

Step 3 runs regardless, so the result is never lost even if 1 and 2 both fail.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SENTINEL = os.path.join(HERE, ".table1_done")
OUT = os.path.join(HERE, "table1_FINAL.txt")
CREDS = os.path.expanduser("~/.config/btp_mail.json")
POLL_SECONDS = 120
MAX_WAIT_HOURS = 24


def build_table() -> str:
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_table1.py")],
                       capture_output=True, text=True, cwd=HERE)
    return r.stdout or r.stderr


def send_email(body: str) -> str:
    if not os.path.exists(CREDS):
        return "skipped (no credentials file)"
    try:
        import smtplib
        from email.message import EmailMessage
        c = json.load(open(CREDS))
        m = EmailMessage()
        m["Subject"] = "PTB-XL Table 1 complete"
        m["From"], m["To"] = c["user"], c.get("to", c["user"])
        m.set_content(body)
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as s:
            s.starttls()
            s.login(c["user"], c["app_password"])
            s.send_message(m)
        return f"sent to {m['To']}"
    except Exception as e:                       # never leak the credential
        return f"failed ({type(e).__name__})"


def desktop(msg: str) -> None:
    try:
        subprocess.run(["notify-send", "BTP: Table 1 complete", msg],
                       timeout=15, capture_output=True)
    except Exception:
        pass


def main() -> None:
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    while not os.path.exists(SENTINEL):
        if time.time() > deadline:
            print("gave up waiting"); return
        time.sleep(POLL_SECONDS)

    table = build_table()
    with open(OUT, "w") as f:
        f.write(table)
    status = send_email(table)
    desktop(f"Table written to {OUT}. Email: {status}")
    print(f"[{time.strftime('%F %T')}] table -> {OUT}; email {status}")


if __name__ == "__main__":
    main()
