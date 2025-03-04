--
-- PostgreSQL database dump
--

-- Dumped from database version 16.1 (Debian 16.1-1.pgdg120+1)
-- Dumped by pg_dump version 16.6 (Ubuntu 16.6-1.pgdg22.04+1)

-- Started on 2025-03-04 10:36:42 UTC

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 3543 (class 0 OID 18251)
-- Dependencies: 216
-- Data for Name: schema; Type: TABLE DATA; Schema: public; Owner: root
--

INSERT INTO public.schema (id, crated_at, is_active, output_template, schema_json_path, schema_type) VALUES (1, 1734009700164, true, NULL, '/data/singlepoint-ai-data/ftw_schema.json', 'FTW_SCHEMA');
INSERT INTO public.schema (id, crated_at, is_active, output_template, schema_json_path, schema_type) VALUES (2, 1734009700164, true, NULL, '/data/singlepoint-ai-data/relius-schema.json', 'RELIUS_SCHEMA');


--
-- TOC entry 3549 (class 0 OID 0)
-- Dependencies: 215
-- Name: schema_id_seq; Type: SEQUENCE SET; Schema: public; Owner: root
--

SELECT pg_catalog.setval('public.schema_id_seq', 1, false);


-- Completed on 2025-03-04 10:36:42 UTC

--
-- PostgreSQL database dump complete
--

