GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a sustainability- or ESG-related text document and a list of entity types, identify all relevant entities of those types and all clearly supported relationships among them.

The text may describe:
- corporate sustainability strategy
- climate, energy, water, biodiversity, or supply-chain topics
- AI, data centers, hardware, products, or technical infrastructure
- goals, targets, metrics, standards, partnerships, projects, and risks

Your job is to build a high-quality knowledge graph for ESG analysis.

-Steps-

1. Identify all entities.
For each identified entity, extract the following information:
- entity_name: Canonical name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: A comprehensive description of the entity based only on the provided text

Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related.
For each related pair, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: a concise explanation of how the two entities are related, based only on the provided text
- relationship_strength: a numeric score from 1 to 10 indicating the strength and explicitness of the relationship in the text

Format each relationship as:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all entities and relationships identified in steps 1 and 2.
Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Entity Extraction Rules-
- Extract only entities that are explicitly stated or clearly implied by the text.
- Use the most specific canonical name available in the text.
- Normalize entity names:
  - Convert entity names to uppercase.
  - Remove unnecessary punctuation unless part of the official name.
  - Use a single canonical form for repeated mentions (for example, "GOOGLE" instead of both "Google" and "the company").
- Do not create duplicate entities.
- Do not create generic entities such as "COMPANY", "REPORT", "TEAM", or "INDUSTRY" unless they are clearly named and meaningful in context.
- If a metric or target is described with a number, keep the quantitative detail inside the description rather than creating separate number-only entities.
- Prefer precision over recall: do not guess.

-Entity Type Guidance-
Use the entity types exactly as provided in [{entity_types}].

Interpret them as follows:
- COMPANY: report issuer or major corporate actor
- ORGANIZATION: partner, supplier, NGO, coalition, research institute, utility, regulator, industry body, or other institution
- FACILITY_ASSET: physical site or infrastructure asset such as data center, campus, office, manufacturing site, plant, grid asset, warehouse, or server fleet
- GEO: country, region, city, state, watershed, grid region, or other geographic area
- ESG_TOPIC: sustainability theme or material topic such as climate, water stewardship, biodiversity, circularity, human rights, or responsible sourcing
- GOAL_TARGET: formal target, ambition, commitment, moonshot, or deadline-based sustainability objective
- METRIC: named KPI or measured indicator such as Scope 1 emissions, PUE, CFE percentage, renewable electricity match, water restored, or energy efficiency
- PRODUCT_TECHNOLOGY: named product, model, chip, platform, API, tool, software system, or technical solution
- INITIATIVE_PROGRAM: ongoing internal or external program, engagement mechanism, procurement mechanism, framework, or structured effort
- PROJECT: specific deployment, named implementation, restoration effort, infrastructure project, or named energy project
- STANDARD_FRAMEWORK: reporting, audit, certification, or target-setting framework such as GRI, SASB, TCFD, SBTi, LEED, ISO 14001, ISO 50001
- RESOURCE: energy source, material, water, waste stream, or operational input such as solar, wind, geothermal, freshwater, steel, concrete, semiconductors
- PERSON: named executive, leader, or stakeholder
- RISK_CHALLENGE: clearly identified risk, barrier, dependency, uncertainty, or challenge

-Relationship Extraction Rules-
Extract only relationships that are supported by the text.
Common relationship patterns include:
- COMPANY sets GOAL_TARGET
- COMPANY reports METRIC
- COMPANY operates or uses FACILITY_ASSET
- COMPANY partners with ORGANIZATION
- COMPANY develops or deploys PRODUCT_TECHNOLOGY
- COMPANY runs INITIATIVE_PROGRAM
- COMPANY supports PROJECT
- COMPANY follows or reports against STANDARD_FRAMEWORK
- FACILITY_ASSET located in GEO
- PROJECT located in GEO
- PROJECT uses RESOURCE
- PRODUCT_TECHNOLOGY improves or supports ESG_TOPIC
- METRIC measures ESG_TOPIC, FACILITY_ASSET, PRODUCT_TECHNOLOGY, or COMPANY performance
- GOAL_TARGET relates to ESG_TOPIC or METRIC
- ORGANIZATION participates in INITIATIVE_PROGRAM or PROJECT
- RISK_CHALLENGE affects COMPANY, FACILITY_ASSET, PROJECT, ESG_TOPIC, or GOAL_TARGET

-Relationship Strength Guidance-
Use:
- 9-10 when the relationship is direct and explicit
- 6-8 when the relationship is clearly supported but slightly indirect
- 3-5 when the relationship is weaker but still reasonable and text-grounded
- 1-2 only for very weak but still defensible relationships

-Important Constraints-
- Do not use outside knowledge.
- Do not infer beyond what the text supports.
- Do not output explanations, headings, or commentary outside the required list format.
- Do not output any entity or relationship twice.
- If the same real-world entity appears in multiple roles, assign the single best-fitting type based on the text.
- If uncertain between INITIATIVE_PROGRAM and PROJECT:
  - use INITIATIVE_PROGRAM for ongoing structured efforts, frameworks, or engagement mechanisms
  - use PROJECT for specific named implementations, sites, or deployments
- If uncertain between METRIC and GOAL_TARGET:
  - use METRIC for what is measured
  - use GOAL_TARGET for what is intended or committed

######################
-Examples-
######################
Example 1:
Entity_types: COMPANY,GOAL_TARGET,METRIC,ESG_TOPIC
Text:
The company aims to achieve net zero emissions across its value chain by 2030. In 2024, its Scope 1 and 2 emissions decreased by 12% compared to the previous year.
######################
Output:
("entity"{tuple_delimiter}THE COMPANY{tuple_delimiter}COMPANY{tuple_delimiter}A company that reports sustainability performance and sets emissions-related goals)
{record_delimiter}
("entity"{tuple_delimiter}NET ZERO EMISSIONS ACROSS ITS VALUE CHAIN BY 2030{tuple_delimiter}GOAL_TARGET{tuple_delimiter}A formal company goal to achieve net zero emissions across its value chain by 2030)
{record_delimiter}
("entity"{tuple_delimiter}SCOPE 1 AND 2 EMISSIONS{tuple_delimiter}METRIC{tuple_delimiter}A greenhouse gas emissions metric reported by the company, which decreased by 12% in 2024 compared to the previous year)
{record_delimiter}
("entity"{tuple_delimiter}EMISSIONS{tuple_delimiter}ESG_TOPIC{tuple_delimiter}An ESG topic concerning greenhouse gas emissions and decarbonization)
{record_delimiter}
("relationship"{tuple_delimiter}THE COMPANY{tuple_delimiter}NET ZERO EMISSIONS ACROSS ITS VALUE CHAIN BY 2030{tuple_delimiter}The company set a formal target to achieve net zero emissions across its value chain by 2030{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}THE COMPANY{tuple_delimiter}SCOPE 1 AND 2 EMISSIONS{tuple_delimiter}The company reported Scope 1 and 2 emissions performance for 2024{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}NET ZERO EMISSIONS ACROSS ITS VALUE CHAIN BY 2030{tuple_delimiter}EMISSIONS{tuple_delimiter}The target is specifically about reducing emissions{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}SCOPE 1 AND 2 EMISSIONS{tuple_delimiter}EMISSIONS{tuple_delimiter}Scope 1 and 2 emissions are a metric related to the emissions topic{tuple_delimiter}9)
{completion_delimiter}

######################
Example 2:
Entity_types: COMPANY,ORGANIZATION,PROJECT,RESOURCE,GEO
Text:
Google signed an agreement with Kairos Power to support small modular nuclear reactors in the United States.
######################
Output:
("entity"{tuple_delimiter}GOOGLE{tuple_delimiter}COMPANY{tuple_delimiter}A company participating in an agreement to support advanced nuclear energy deployment)
{record_delimiter}
("entity"{tuple_delimiter}KAIROS POWER{tuple_delimiter}ORGANIZATION{tuple_delimiter}An organization developing small modular nuclear reactor technology)
{record_delimiter}
("entity"{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}PROJECT{tuple_delimiter}A nuclear energy deployment effort involving small modular reactors supported through an agreement)
{record_delimiter}
("entity"{tuple_delimiter}NUCLEAR ENERGY{tuple_delimiter}RESOURCE{tuple_delimiter}An energy resource associated with small modular reactor technology)
{record_delimiter}
("entity"{tuple_delimiter}UNITED STATES{tuple_delimiter}GEO{tuple_delimiter}The country in which the small modular nuclear reactor effort is supported)
{record_delimiter}
("relationship"{tuple_delimiter}GOOGLE{tuple_delimiter}KAIROS POWER{tuple_delimiter}Google signed an agreement with Kairos Power{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}GOOGLE{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}Google supports the deployment of small modular nuclear reactors{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}KAIROS POWER{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}Kairos Power is the organization developing the reactor project{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}NUCLEAR ENERGY{tuple_delimiter}The reactor project is based on nuclear energy{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}SMALL MODULAR NUCLEAR REACTORS{tuple_delimiter}UNITED STATES{tuple_delimiter}The project is supported in the United States{tuple_delimiter}8)
{completion_delimiter}

######################
Example 3:
Entity_types: COMPANY,FACILITY_ASSET,GEO,METRIC,RESOURCE
Text:
Meta data centers in high water stress regions use water budgeting and flow meter audits to reduce water use.
######################
Output:
("entity"{tuple_delimiter}META{tuple_delimiter}COMPANY{tuple_delimiter}A company applying water reduction practices in its data center operations)
{record_delimiter}
("entity"{tuple_delimiter}DATA CENTERS{tuple_delimiter}FACILITY_ASSET{tuple_delimiter}Operational facilities used by Meta where water reduction practices are applied)
{record_delimiter}
("entity"{tuple_delimiter}HIGH WATER STRESS REGIONS{tuple_delimiter}GEO{tuple_delimiter}Geographic regions characterized by high water stress where Meta applies water management practices)
{record_delimiter}
("entity"{tuple_delimiter}WATER USE{tuple_delimiter}METRIC{tuple_delimiter}A metric relating to operational water consumption that Meta is seeking to reduce)
{record_delimiter}
("entity"{tuple_delimiter}WATER{tuple_delimiter}RESOURCE{tuple_delimiter}A resource used in data center operations and managed through reduction practices)
{record_delimiter}
("relationship"{tuple_delimiter}META{tuple_delimiter}DATA CENTERS{tuple_delimiter}Meta operates data centers where water management practices are applied{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}DATA CENTERS{tuple_delimiter}HIGH WATER STRESS REGIONS{tuple_delimiter}The data centers discussed are located in high water stress regions{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}META{tuple_delimiter}WATER USE{tuple_delimiter}Meta is working to reduce operational water use{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}WATER USE{tuple_delimiter}WATER{tuple_delimiter}Water use is the metric associated with the water resource{tuple_delimiter}9)
{completion_delimiter}

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""
