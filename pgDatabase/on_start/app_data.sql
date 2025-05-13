DROP TRIGGER IF EXISTS trigger_update_updated_at ON thrive_user;
DROP TRIGGER IF EXISTS trigger_update_updated_at ON thrive_message;

DROP TABLE IF EXISTS thrive_message;
DROP TABLE IF EXISTS thrive_user;
DROP TABLE IF EXISTS thrive_user_role;

CREATE TABLE thrive_user_role (
    id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE thrive_user (
    id SERIAL PRIMARY KEY,
	user_role_id INTEGER REFERENCES thrive_user_role(id),
    username VARCHAR(50) NOT NULL UNIQUE,
	first_name VARCHAR(50) NOT NULL,
	last_name VARCHAR(50) NOT NULL,
    --email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	show_sql BOOLEAN DEFAULT false,
	show_table BOOLEAN DEFAULT false,
	show_plotly_code BOOLEAN DEFAULT false,
	show_chart BOOLEAN DEFAULT false,
	show_question_history BOOLEAN DEFAULT false,
	show_summary BOOLEAN DEFAULT false,
	voice_input BOOLEAN DEFAULT false,
	speak_summary BOOLEAN DEFAULT false,
	show_suggested BOOLEAN DEFAULT false,
	show_followup BOOLEAN DEFAULT false,
	show_elapsed_time BOOLEAN DEFAULT false,
	llm_fallback BOOLEAN DEFAULT false,
	min_message_id INTEGER DEFAULT 0,
	enable_sql_retries BOOLEAN DEFAULT false
);

--create a table to house all of the message data...
CREATE TABLE thrive_message (
	id SERIAL PRIMARY KEY,
	user_id INTEGER REFERENCES thrive_user(id),
	role VARCHAR(50) NOT NULL,
	content TEXT NOT NULL,
	type VARCHAR(50) NOT NULL,
	feedback VARCHAR(50),
	query TEXT,
	question VARCHAR(1000),
	dataframe Text,
	elapsed_time NUMERIC(10, 6),
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_updated_at
BEFORE UPDATE ON thrive_user
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_updated_at
BEFORE UPDATE ON thrive_message
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

INSERT INTO thrive_user_role (role_name, description) VALUES ('Admin', 'Administrator with full access');
INSERT INTO thrive_user_role (role_name, description) VALUES ('Doctor', 'A physician who has the rights to view some individual patient data');
INSERT INTO thrive_user_role (role_name, description) VALUES ('Patient', 'Patient access, only has access to see their own data or population data');

INSERT INTO thrive_user (username, first_name, last_name, show_summary, password, user_role_id) 
VALUES ('thriveai-kr', 'Kyle', 'Root', true, '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
        (SELECT id FROM thrive_user_role WHERE role_name = 'Patient'));
INSERT INTO thrive_user (username, first_name, last_name, show_summary, password, user_role_id) 
VALUES ('thriveai-je', 'Joseph', 'Eberle', true, '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
        (SELECT id FROM thrive_user_role WHERE role_name = 'Patient'));
INSERT INTO thrive_user (username, first_name, last_name, show_summary, password, user_role_id) 
VALUES ('thriveai-as', 'Al', 'Seoud', true, '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
        (SELECT id FROM thrive_user_role WHERE role_name = 'Patient'));
INSERT INTO thrive_user (username, first_name, last_name, show_summary, password, user_role_id) 
VALUES ('thriveai-fm', 'Frankly', 'Metty', true, '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
        (SELECT id FROM thrive_user_role WHERE role_name = 'Patient'));
INSERT INTO thrive_user (username, first_name, last_name, show_summary, password, user_role_id) 
VALUES ('thriveai-dr', 'Dr.', 'Smith', true, '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
        (SELECT id FROM thrive_user_role WHERE role_name = 'Doctor'));
		INSERT INTO thrive_user (username, first_name, last_name, show_summary, password, user_role_id) 
VALUES ('thriveai-re', 'Rob', 'Enderle', true, '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 
        (SELECT id FROM thrive_user_role WHERE role_name = 'Patient'));
