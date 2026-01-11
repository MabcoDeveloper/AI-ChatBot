"# AI-ChatBot"  

## Notes on conversation logging and ML

- Conversations are logged to a MongoDB collection named `training_examples` when the feature is enabled.
- Each session is assigned a `session_id` and turns are appended; sessions are labeled when outcomes occur (e.g., `purchase_completed`, `purchase_cancelled`, `session_ended`).
- Respect user privacy: don't enable logging in production without informing users and obtaining consent.
