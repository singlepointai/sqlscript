--
-- PostgreSQL database dump
--

-- Dumped from database version 16.1 (Debian 16.1-1.pgdg120+1)
-- Dumped by pg_dump version 16.6 (Ubuntu 16.6-1.pgdg22.04+1)

-- Started on 2025-03-04 10:46:59 UTC

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
-- TOC entry 3542 (class 0 OID 18271)
-- Dependencies: 220
-- Data for Name: schema_to_data_dictionary; Type: TABLE DATA; Schema: public; Owner: root
--

INSERT INTO public.schema_to_data_dictionary (id, data_dictionary, schema) VALUES (1, 1, 1);
INSERT INTO public.schema_to_data_dictionary (id, data_dictionary, schema) VALUES (2, 2, 1);
INSERT INTO public.schema_to_data_dictionary (id, data_dictionary, schema) VALUES (3, 1, 2);


--
-- TOC entry 3548 (class 0 OID 0)
-- Dependencies: 219
-- Name: schema_to_data_dictionary_id_seq; Type: SEQUENCE SET; Schema: public; Owner: root
--

SELECT pg_catalog.setval('public.schema_to_data_dictionary_id_seq', 1, false);


-- Completed on 2025-03-04 10:46:59 UTC

--
-- PostgreSQL database dump complete
--

