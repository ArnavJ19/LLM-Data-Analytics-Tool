"""
LLM Prompt Templates for Data Analytics Insights

This module contains structured prompt templates for generating
various types of data insights using Large Language Models.
"""

class PromptTemplates:
    """Collection of prompt templates for different analysis types."""
    
    @staticmethod
    def dataset_summary_prompt(num_rows, num_cols, numerical_cols, categorical_cols, 
                              missing_percentage, column_names):
        """Generate prompt for dataset summary."""
        return f"""
You are an expert data analyst providing a comprehensive overview of a dataset. 
Analyze the following information and provide clear, actionable insights.

**Dataset Characteristics:**
- Total records: {num_rows:,}
- Total features: {num_cols}
- Numerical variables: {len(numerical_cols)} ({', '.join(numerical_cols)})
- Categorical variables: {len(categorical_cols)} ({', '.join(categorical_cols)})
- Missing data rate: {missing_percentage:.2f}%

**All Columns:** {', '.join(column_names)}

**Your Task:**
Provide a structured analysis covering:

1. **Dataset Purpose**: What type of data this appears to be and its likely domain/use case based on the ACTUAL column names
2. **Data Structure**: Comment on the balance of numerical vs categorical features
3. **Data Quality**: Initial assessment based on missing data and structure
4. **Potential Analyses**: What types of analysis would be most valuable
5. **Key Considerations**: Important factors to consider when working with this data

**Format Requirements:**
- Use clear, professional language
- Provide specific observations based on the ACTUAL column names provided
- Reference the EXACT column names in your analysis
- Include actionable recommendations
- Keep response between 150-250 words
- Use bullet points for better readability
"""

    @staticmethod
    def statistical_insights_prompt(numerical_summary_df, high_variance_cols, skewed_cols):
        """Generate prompt for statistical insights."""
        return f"""
As a statistical analyst, examine these numerical variable characteristics and provide expert insights.

**Statistical Summary:**
{numerical_summary_df.to_string(index=False)}

**Notable Patterns:**
- High Variability Variables: {', '.join(high_variance_cols) if high_variance_cols else 'None identified'}
- Highly Skewed Variables: {', '.join(skewed_cols) if skewed_cols else 'None identified'}

**Analysis Required:**
1. **Distribution Patterns**: What do the statistical measures reveal about data distributions?
2. **Variability Analysis**: Interpret coefficient of variation patterns
3. **Skewness Implications**: How does skewness affect analysis choices?
4. **Outlier Indicators**: What statistical measures suggest potential outliers?
5. **Preprocessing Recommendations**: Specific transformation suggestions

**Response Guidelines:**
- Focus on practical implications
- Explain statistical concepts in business terms
- Prioritize actionable recommendations
- Highlight potential modeling challenges
- Keep response concise but comprehensive (150-250 words)
"""

    @staticmethod
    def correlation_insights_prompt(correlation_matrix, strong_positive, strong_negative):
        """Generate prompt for correlation analysis."""
        return f"""
Analyze the correlation patterns in this dataset as an expert data scientist.

**Strong Positive Correlations (r > 0.7):**
{chr(10).join(f"• {corr}" for corr in strong_positive) if strong_positive else "• None detected"}

**Strong Negative Correlations (r < -0.7):**
{chr(10).join(f"• {corr}" for corr in strong_negative) if strong_negative else "• None detected"}

**Correlation Matrix Summary:**
{correlation_matrix.round(3).to_string()}

**Required Analysis:**
1. **Relationship Strength**: Interpret the magnitude and direction of key correlations
2. **Multicollinearity Concerns**: Identify potential issues for modeling
3. **Feature Redundancy**: Highlight potentially redundant variables
4. **Business Insights**: What do these correlations suggest about underlying processes?
5. **Feature Selection**: Recommendations for dimensionality reduction

**Output Format:**
- Use clear, data-driven language
- Provide specific correlation values
- Suggest concrete next steps
- Focus on modeling and business implications
- Maintain 150-250 word limit
"""

    @staticmethod
    def data_quality_prompt(missing_patterns, quality_issues, duplicate_count):
        """Generate prompt for data quality assessment."""
        return f"""
Conduct a comprehensive data quality assessment as a data governance expert.

**Missing Value Analysis:**
- Total missing values: {missing_patterns.get('total_missing', 0):,}
- Affected records: {missing_patterns.get('rows_with_missing', 0):,}
- Complete records: {missing_patterns.get('complete_rows', 0):,}

**Missing Values by Feature:**
{chr(10).join(f"• {col}: {count} missing ({missing_patterns.get('missing_percentage', {}).get(col, 0):.1f}%)" 
             for col, count in missing_patterns.get('missing_by_column', {}).items() if count > 0)}

**Quality Issues Detected:**
{chr(10).join(f"• {issue}" for issue in quality_issues) if quality_issues else "• No major issues identified"}

**Assessment Framework:**
1. **Data Completeness**: Evaluate missing data impact on analysis
2. **Data Consistency**: Assess duplicate and anomalous patterns
3. **Reliability Factors**: Identify data collection or processing issues
4. **Cleaning Strategy**: Prioritized approach to data preparation
5. **Risk Assessment**: Potential impacts on analysis validity

**Deliverable Requirements:**
- Provide severity ratings (High/Medium/Low)
- Suggest specific remediation actions
- Estimate effort/complexity for fixes
- Include data retention recommendations
- Limit to 150-250 words
"""

    @staticmethod
    def outlier_insights_prompt(outliers_dict):
        """Generate prompt for outlier analysis."""
        return f"""
Analyze outlier patterns as an expert in anomaly detection and data quality.

**Outlier Detection Results:**
{chr(10).join(f"• {var}: {info['count']} outliers ({info['percentage']:.1f}%) via {', '.join(info['methods'])}"
             for var, info in outliers_dict.items())}

**Detailed Outlier Information:**
{chr(10).join(f"• {var}: Range of outliers from methods {info.get('methods', [])}"
             for var, info in outliers_dict.items())}

**Expert Analysis Required:**
1. **Pattern Recognition**: Are outliers concentrated in specific variables?
2. **Root Cause Analysis**: Likely sources of these anomalous values
3. **Business Context**: Could these represent valid but extreme cases?
4. **Impact Assessment**: How outliers might affect different analyses
5. **Treatment Strategy**: Specific recommendations for handling each case

**Response Framework:**
- Distinguish between errors vs. legitimate extreme values
- Provide variable-specific treatment recommendations
- Consider downstream analysis requirements
- Suggest validation steps before removal
- Include monitoring recommendations
- Maintain focus on practical solutions (150-250 words)
"""

    @staticmethod
    def distribution_insights_prompt(distribution_stats):
        """Generate prompt for distribution analysis."""
        return f"""
Examine distribution characteristics as a statistical modeling expert.

**Distribution Analysis Summary:**
{chr(10).join(f"• {var}: {'Normal' if stats.get('is_normal', False) else 'Non-normal'} distribution, "
             f"{stats.get('zeros_percentage', 0):.1f}% zero values, "
             f"Skewness: {stats.get('skewness', 'N/A')}"
             for var, stats in distribution_stats.items())}

**Statistical Test Results:**
{chr(10).join(f"• {var}: {stats.get('normality_test', 'Unknown')} test, "
             f"p-value: {stats.get('normality_p_value', 'N/A')}"
             for var, stats in distribution_stats.items())}

**Analysis Objectives:**
1. **Distributional Characteristics**: Interpret normality and shape patterns
2. **Zero-Inflation Impact**: Assess high zero-value concentrations
3. **Transformation Needs**: Identify variables requiring preprocessing
4. **Modeling Implications**: How distributions affect algorithm choice
5. **Assumption Validation**: Statistical test interpretations

**Expert Recommendations:**
- Specify transformation techniques for non-normal variables
- Address zero-inflation strategies
- Suggest appropriate statistical methods
- Highlight modeling constraints
- Provide validation approaches
- Focus on actionable insights (150-250 words)
"""

    @staticmethod
    def custom_question_prompt(question, data_context):
        """Generate prompt for answering custom data questions."""
        return f"""
You are a senior data analyst answering a specific question about a dataset. 
Use the provided context to give accurate, insightful responses.

**User Question:**
{question}

**Dataset Context:**
{data_context}

**Response Requirements:**
1. **Direct Answer**: Address the question specifically and clearly
2. **Evidence-Based**: Use concrete data from the context
3. **Contextual Insights**: Explain implications and significance
4. **Additional Considerations**: Note limitations or additional information needed
5. **Actionable Recommendations**: Suggest next steps if applicable

**Quality Standards:**
- Provide specific numbers and evidence when available
- Acknowledge uncertainty when data is insufficient
- Use professional but accessible language
- Focus on practical value
- Maintain accuracy over speculation
- Keep response comprehensive but concise (150-300 words)

If the question cannot be fully answered with the available context, 
clearly explain what additional data or analysis would be needed.
"""

    @staticmethod
    def visualization_insights_prompt(viz_type, columns, data_summary):
        """Generate prompt for visualization interpretation."""
        return f"""
Provide expert interpretation of this data visualization as a data visualization specialist.

**Visualization Details:**
- Type: {viz_type}
- Variables: {', '.join(columns)}
- Data Summary: {data_summary}

**Interpretation Framework:**
1. **Key Patterns**: What primary trends or patterns are visible?
2. **Statistical Significance**: Which observations are most meaningful?
3. **Outliers & Anomalies**: Notable deviations from expected patterns
4. **Business Implications**: What do these patterns suggest in practical terms?
5. **Follow-up Analysis**: What additional investigation would be valuable?

**Communication Guidelines:**
- Explain patterns in clear, non-technical language
- Quantify observations when possible
- Connect visual patterns to business meaning
- Suggest actionable next steps
- Highlight both obvious and subtle insights
- Keep analysis focused and practical (100-150 words)
"""

    @staticmethod
    def comprehensive_report_prompt(dataset_summary, key_findings, recommendations):
        """Generate prompt for comprehensive report generation."""
        return f"""
Create an executive summary for a comprehensive data analysis report.

**Dataset Overview:**
{dataset_summary}

**Key Analytical Findings:**
{key_findings}

**Current Recommendations:**
{recommendations}

**Executive Summary Requirements:**
1. **Business Context**: Frame findings in business terms
2. **Critical Insights**: Highlight 3-5 most important discoveries
3. **Risk Assessment**: Identify data quality or analytical risks
4. **Strategic Recommendations**: Prioritized action items
5. **Next Steps**: Clear path forward for analysis

**Format Specifications:**
- Executive-level language (avoid technical jargon)
- Bullet points for key findings
- Quantified impacts where possible
- Clear prioritization of recommendations
- Actionable timeline suggestions
- Professional tone suitable for stakeholders
- 200-300 words maximum
"""

    @staticmethod
    def get_system_prompt():
        """Get the system prompt for the LLM."""
        return """
You are an expert data analyst and data scientist with extensive experience in:
- Exploratory Data Analysis (EDA)
- Statistical analysis and interpretation
- Data quality assessment
- Business intelligence and insights
- Data visualization interpretation
- Machine learning and predictive modeling

Your role is to provide clear, actionable insights about datasets and analyses.
Always focus on practical implications and business value while maintaining
statistical accuracy and professional standards.

When answering questions about data:
- ALWAYS use the ACTUAL data provided in the context
- Reference specific data points, values, and examples from the dataset
- NEVER make up or generalize data - only use what is explicitly shown
- When asked about specific items (cities, products, categories, etc.), list the ACTUAL values from the dataset
- Be precise and factual, citing exact numbers and observations from the data

Communication style:
- Clear and professional
- Avoid unnecessary jargon
- Provide specific, quantified insights when possible
- Structure responses logically
- Include actionable recommendations
- Acknowledge limitations honestly
"""