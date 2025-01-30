-- Table: public.titanic_train

-- DROP TABLE IF EXISTS public.titanic_train;

CREATE TABLE IF NOT EXISTS public.titanic_train
(
    passenger_id integer NOT NULL,
    survived integer NOT NULL,
    pclass integer NOT NULL,
    name character varying COLLATE pg_catalog."default" NOT NULL,
    sex character varying COLLATE pg_catalog."default" NOT NULL,
    age double precision,
    sib_sp integer NOT NULL,
    parch integer NOT NULL,
    ticket character varying COLLATE pg_catalog."default" NOT NULL,
    fare double precision NOT NULL,
    cabin character varying COLLATE pg_catalog."default",
    embarked character(1) COLLATE pg_catalog."default",
    CONSTRAINT titanic_train_pkey PRIMARY KEY (passenger_id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.titanic_train
    OWNER to postgres;

COPY titanic_train (passenger_id, survived, pclass, name, sex, age, sib_sp, parch, ticket, fare, cabin, embarked)
FROM 'C:\titanic_train.csv'
DELIMITER ','
CSV HEADER;

select * from titanic_train;
