-- Table: public.penguins

-- DROP TABLE IF EXISTS public.penguins;

CREATE TABLE IF NOT EXISTS public.penguins
(
    "species" character varying NOT NULL,
    "island" character varying NOT NULL,
    "bill_length_mm" character varying NOT NULL,
    "bill_depth_mm" character varying NOT NULL,
    "flipper_length_mm" character varying NOT NULL,
    "body_mass_g" character varying NOT NULL,
    "sex" character varying NOT NULL,
    "year" integer NOT NULL
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.penguins
    OWNER to postgres;


COPY penguins ("species" ,"island" ,"bill_length_mm" ,"bill_depth_mm" ,"flipper_length_mm" ,"body_mass_g" ,"sex" ,"year")
FROM 'C:\penguins.csv'
DELIMITER ','
CSV HEADER;

select * from public.penguins;