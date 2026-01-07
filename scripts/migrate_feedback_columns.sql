-- Migration for feedback columns (since Jan 6, 2025)
-- Run these commands in your SQLite database

ALTER TABLE thrive_message ADD COLUMN feedback_comment VARCHAR(500);
ALTER TABLE thrive_message ADD COLUMN training_status VARCHAR(20);
ALTER TABLE thrive_message ADD COLUMN reviewed_by INTEGER REFERENCES thrive_user(id);
ALTER TABLE thrive_message ADD COLUMN reviewed_at TIMESTAMP;
