CHAPTER 1
=========

Data
----

.. admonition:: Data Warehouses

   - Definition:
      - Centrallized repository for structured data, optimized for complex queries and analysis.
      - Use ETL (Extract -> Transform -> Load) to clean and structure data before storage.

   - Key Features:
      - Schema-on-write: Schema is defined before data is written.
      - Store data in star or snowflake schemas.
      - Optimized for read-heavy workloads (e.g. business intelligence)
      - Typically contains pre-processed, cleaned data.

   - Use Cases:
      - Business intelligence, reporting, dashboards.
      - Fast, reliable analytical queries on known data.

   - AWS Example:
      - Amazon Redshift (primary data warehouse offering)

.. admonition:: Data Lakes

   - Definition:
      - Large-scale repository for raw data in its native format.
      - Supports structured, semi-structured, and unstructured data.
      - Uses ELT (Extract -> Load -> Transform).

   - Key Features:
      - Schema-on-read: Structure is interpreted onlly when the data is read.
      - Highly flexible: Store anything without worrying about schema first.
      - Suitable for batch, real-time, and stream processing.

   - Use Cases:
      - Data science, machine learning, exploratory analysis.
      - Collecting logs, clickstream data, IoT data, etc.

   - AWS Example:
      - Amazon S3 used as a data lake.
      - AWS Glue: catalog and extract schema.
      - Amazon Athena: query raw data using SQL.


.. admonition:: ✅ Comparison: Data Warehouse vs Data Lake

   +--------------------+-------------------------+--------------------------------------------+
   | Feature            | Data Warehouse          | Data Lake                                  |
   +====================+=========================+============================================+
   | Schema             | Schema-on-write         | Schema-on-read                             |
   +--------------------+-------------------------+--------------------------------------------+
   | Data Types         | Structured only         | Structured + semi + unstructured           |
   +--------------------+-------------------------+--------------------------------------------+
   | Data Ingestion     | ETL                     | ELT                                        |
   +--------------------+-------------------------+--------------------------------------------+
   | Storage Cost       | Higher                  | Lower                                      |
   +--------------------+-------------------------+--------------------------------------------+
   | Query Optimization | Highly optimized        | Less optimized                             |
   +--------------------+-------------------------+--------------------------------------------+
   | Flexibility        | Rigid (hard to change)  | Flexible                                   |
   +--------------------+-------------------------+--------------------------------------------+
   | Tools (AWS)        | Amazon Redshift         | S3 + Glue + Athena                         |
   +--------------------+-------------------------+--------------------------------------------+
   | Use Case           | BI & reporting          | ML, data discovery, raw storage            |
   +--------------------+-------------------------+--------------------------------------------+

.. admonition:: Data Lakehouse

   - Definition:
      - Hybrid architecture combining features of data lake and data warehouses.
      - Provides low-cost storage + structured querying and reliablility.

   - Key Features:
      - Supports both schema-on-read and schema-on-write.
      - Unified support for analytics and machine learning.
      - Built on distributed / cloud systems.

   - AWS Examples:
      - S3 + AWS Lake Formation + Redshift Spectrum.
      - Redshift Spectrum lets you query S3 raw data using SQL.

   - Other Ecosystem Examples (non- AWS):
      - Delta Lake (Databricks), Apache Hudi, Apache Iceberg, Azure Synapse.

.. admonition:: ✅ When to Use What?

   +-----------------------------------------------+------------------------+
   | Use Case                                      | Recommended Solution   |
   +===============================================+========================+
   | Known structure, fast queries, BI focus       | Data Warehouse         |
   +-----------------------------------------------+------------------------+
   | Mixed/raw data, flexible analytics            | Data Lake              |
   +-----------------------------------------------+------------------------+
   | Need both: BI + ML + flexibility              | Data Lakehouse         |
   +-----------------------------------------------+------------------------+

.. admonition:: Data Mesh

   .. image:: images/data_1.png

   - Definition:

      - A data mesh is an organizational approach to data management, not a specific technology.

      - Focuses on decentralized ownership of data and domain-based data management.

   - Key Principles:

      - Domain Ownership:

         - Data is owned by the teams (or domains) that are closest to it.

         - Each domain is responsible for maintaining and serving its data as a data product.

      - Data as a Product:

         - Teams treat their data as a product that others can discover and use.

         - This promotes reusability, quality, and discoverability.

      - Self-Service Infrastructure:

         - Teams are supported with tools and infrastructure to publish and manage their data.

         - This includes data lakes, data warehouses, and ETL tools.

      - Federated Governance:

         - While data is decentralized, central governance standards ensure:

            - Security

            - Access control

            - Compliance

            - Data quality

   - AWS Tools in Data Mesh Implementation:

      - S3 – Storage for raw and curated data.

      - Lake Formation – For managing permissions, organizing, and securing data lakes.

      - AWS Glue – For data cataloging and ETL.

      - Athena – For querying data products in S3.

      - IAM – For federated access control.

      - Quicksight – For consuming and visualizing data products.

   - Why It Matters:

      - Promotes scalability by allowing teams to manage their own data.

      - Encourages agility in analytics and data-driven decision making.

      - Inspired by large-scale organizations like Amazon, which have long used decentralized models.



