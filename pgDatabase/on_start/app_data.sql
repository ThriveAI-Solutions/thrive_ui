DROP TRIGGER IF EXISTS trigger_update_updated_at ON thrive_user;
DROP TRIGGER IF EXISTS trigger_update_updated_at ON thrive_message;

DROP TABLE IF EXISTS thrive_message;
DROP TABLE IF EXISTS thrive_user;
DROP TABLE IF EXISTS thrive_user_role;
