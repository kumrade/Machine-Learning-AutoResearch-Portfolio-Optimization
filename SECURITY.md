# Security Policy

## Broker credentials

Never commit API keys, client IDs, passwords, PINs, TOTP seeds, session tokens
or screenshots displaying them. Use runtime entry or environment variables.

If credentials have appeared in a screenshot, chat, commit or public repository:

1. Revoke or rotate them immediately through the broker.
2. Remove the exposed file from the repository and its Git history.
3. Review account sessions and trading activity.
4. Replace the credential with a new value stored outside source control.

This project does not require credentials when using synthetic demonstration data.

