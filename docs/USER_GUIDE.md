# Thrive AI User Guide

Welcome to Thrive AI - your intelligent data analysis companion! This guide will help you get started and make the most of the platform's powerful features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Login and Authentication](#login-and-authentication)
3. [Chat Interface Overview](#chat-interface-overview)
4. [Asking Questions About Your Data](#asking-questions-about-your-data)
5. [Magic Commands](#magic-commands)
6. [Data Analysis Workflows](#data-analysis-workflows)
7. [User Settings](#user-settings)
8. [Voice Features](#voice-features)
9. [Tips and Best Practices](#tips-and-best-practices)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### What is Thrive AI?

Thrive AI is an intelligent data analysis platform that lets you:
- Ask questions about your data in plain English
- Generate SQL queries automatically
- Create visualizations and statistical analyses
- Perform advanced analytics with simple commands
- Export and share your findings

### First Time Access

1. **Open the Application**: Navigate to `http://localhost:8501` in your web browser
2. **Choose Your Account**: Use one of the default accounts provided by your administrator
3. **Change Your Password**: Immediately change your password after first login for security

## Login and Authentication

### Default User Accounts

Your administrator has set up the following accounts:

| Username | Role | Default Password |
|----------|------|------------------|
| thriveai-kr | Admin | password |
| thriveai-je | Admin | password |
| thriveai-as | Admin | password |
| thriveai-fm | Admin | password |
| thriveai-dr | Doctor | password |
| thriveai-re | Admin | password |

### Logging In

1. Enter your username and password
2. Click "Login"
3. You'll be redirected to the main chat interface

### Security Best Practices

- **Change your password immediately** after first login
- Use a strong, unique password
- Log out when finished using the system
- Report any suspicious activity to your administrator

## Chat Interface Overview

### Main Components

The Thrive AI interface consists of several key areas:

#### 1. Navigation Bar
- **Chat Bot** 🤖: Main analysis interface
- **User Settings** 👤: Account management and preferences

#### 2. Chat Area
- **Message History**: Previous conversations and analyses
- **Input Box**: Where you type questions and commands
- **Send Button**: Submit your questions

#### 3. Sidebar (Chat Bot Page)
- **Database Settings**: Connection and configuration options
- **User Preferences**: Customize your experience
- **Voice Controls**: Enable speech features

### Getting Oriented

When you first log in, you'll see:
- A welcome message
- Sample questions to get you started
- The available databases and tables
- Recent conversation history (if any)

## Asking Questions About Your Data

### Natural Language Queries

Thrive AI understands natural language questions about your data. Here are examples:

#### Basic Questions
```
"How many records are in the penguins table?"
"What are the different species in the penguins dataset?"
"Show me the average bill length by species"
```

#### More Complex Queries
```
"What's the correlation between bill length and body mass for Adelie penguins?"
"Show me the distribution of ages in the titanic dataset"
"Which species has the highest average flipper length?"
```

#### Comparative Analysis
```
"Compare the survival rates between different passenger classes in the titanic data"
"How do penguin measurements differ between islands?"
"What are the trends in health data over time?"
```

### Query Results

When you ask a question, Thrive AI will:
1. **Generate SQL**: Create the appropriate database query
2. **Execute Query**: Run it against your data
3. **Display Results**: Show data in tables, charts, or summaries
4. **Provide Insights**: Offer interpretation and context

### Follow-up Questions

After getting results, you can ask follow-up questions:
```
"Can you show this as a bar chart?"
"What about the same analysis for male penguins only?"
"Export this data to CSV"
```

## Magic Commands

Magic commands are powerful shortcuts for advanced analysis. They start with a forward slash (`/`) and provide specialized functionality.

### Statistical Analysis Commands

#### `/describe <table>`
Get comprehensive descriptive statistics for all numeric columns in a table.

**Example:**
```
/describe penguins
```

**What you'll get:**
- Count, mean, median, standard deviation
- Min/max values and quartiles
- Skewness and kurtosis
- Missing value counts

#### `/distribution <table>.<column>`
Analyze the distribution of a specific column with statistical tests.

**Example:**
```
/distribution penguins.bill_length_mm
```

**What you'll get:**
- Histogram with distribution curve
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Distribution parameters
- Outlier detection

#### `/correlation <table>.<column1>.<column2>`
Detailed correlation analysis between two variables.

**Example:**
```
/correlation penguins.bill_length_mm.body_mass_g
```

**What you'll get:**
- Correlation coefficient and p-value
- Confidence intervals
- Scatter plot with trend line
- Statistical significance interpretation

#### `/outliers <table>.<column>`
Multi-method outlier detection for a numeric column.

**Example:**
```
/outliers penguins.body_mass_g
```

**What you'll get:**
- IQR-based outliers
- Z-score outliers
- Isolation Forest outliers
- Box plot visualization

### Data Quality Commands

#### `/profile <table>`
Comprehensive data profiling report for a table.

**Example:**
```
/profile titanic_train
```

**What you'll get:**
- Column types and statistics
- Missing value analysis
- Unique value counts
- Data quality scores

#### `/missing <table>`
Detailed missing data analysis.

**Example:**
```
/missing titanic_train
```

**What you'll get:**
- Missing value patterns
- Heatmap of missing data
- Percentage of completeness
- Recommendations for handling

#### `/duplicates <table>`
Find and analyze duplicate records.

**Example:**
```
/duplicates penguins
```

**What you'll get:**
- Exact duplicate count
- Duplicate value analysis by column
- Recommendations for data cleaning

### Visualization Commands

#### `/boxplot <table>.<column>`
Create detailed box plots with statistical annotations.

**Example:**
```
/boxplot penguins.flipper_length_mm
```

#### `/heatmap <table>`
Generate correlation heatmaps for numeric columns.

**Example:**
```
/heatmap penguins
```

#### `/wordcloud <table>` or `/wordcloud <table>.<column>`
Create word clouds from text data.

**Example:**
```
/wordcloud titanic_train.name
```

### Machine Learning Commands

#### `/clusters <table>`
Perform K-means clustering analysis.

**Example:**
```
/clusters penguins
```

**What you'll get:**
- Optimal cluster count (elbow method)
- Cluster assignments
- Visualization of clusters
- Cluster characteristics

#### `/pca <table>`
Principal Component Analysis for dimensionality reduction.

**Example:**
```
/pca penguins
```

**What you'll get:**
- Principal components
- Explained variance
- Scree plot
- Component loadings

### Reporting Commands

#### `/report <table>`
Generate a comprehensive data analysis report.

**Example:**
```
/report wny_health
```

**What you'll get:**
- Executive summary
- Data overview and quality assessment
- Statistical analysis
- Key insights and recommendations

#### `/summary <table>`
Quick executive summary of key findings.

**Example:**
```
/summary penguins
```

### Follow-up Commands

After running any query, you can use these commands on the results:

```
/followup describe
/followup heatmap
/followup clusters
```

Or simply use the magic commands directly:
```
describe
heatmap
clusters
```

### Help Commands

- `/help` - Show all available magic commands
- `/followuphelp` - Show follow-up commands for query results

## Data Analysis Workflows

### Workflow 1: Exploring a New Dataset

1. **Start with data profiling:**
   ```
   /profile penguins
   ```

2. **Check data quality:**
   ```
   /missing penguins
   /duplicates penguins
   ```

3. **Get statistical overview:**
   ```
   /describe penguins
   ```

4. **Explore relationships:**
   ```
   /heatmap penguins
   ```

5. **Ask specific questions:**
   ```
   "What are the main differences between penguin species?"
   ```

### Workflow 2: Investigating a Specific Variable

1. **Examine distribution:**
   ```
   /distribution penguins.body_mass_g
   ```

2. **Check for outliers:**
   ```
   /outliers penguins.body_mass_g
   ```

3. **Explore relationships:**
   ```
   /correlation penguins.body_mass_g.flipper_length_mm
   ```

4. **Visualize patterns:**
   ```
   /boxplot penguins.body_mass_g
   ```

### Workflow 3: Comparative Analysis

1. **Ask comparative questions:**
   ```
   "Compare body mass between male and female penguins"
   ```

2. **Follow up with statistical tests:**
   ```
   /followup describe
   ```

3. **Visualize differences:**
   ```
   "Show this as a box plot grouped by sex"
   ```

### Workflow 4: Finding Patterns and Clusters

1. **Perform clustering:**
   ```
   /clusters penguins
   ```

2. **Reduce dimensions:**
   ```
   /pca penguins
   ```

3. **Ask about patterns:**
   ```
   "What characteristics define each cluster?"
   ```

## User Settings

### Accessing User Settings

Click the **User Settings** 👤 tab in the navigation bar to access:

- **Change Password**: Update your account password
- **Display Preferences**: Customize how data is shown
- **Analysis Settings**: Configure default analysis parameters
- **Export Options**: Set default export formats

### Customizing Your Experience

#### Display Settings
- **Theme**: Choose light or dark mode
- **Chart Style**: Select default visualization style
- **Results per Page**: Set how many results to show at once

#### Analysis Preferences
- **Significance Level**: Set default p-value threshold (usually 0.05)
- **Confidence Interval**: Choose confidence level (90%, 95%, 99%)
- **Precision**: Number of decimal places in results

## Voice Features

### Enabling Voice

1. Go to the sidebar on the Chat Bot page
2. Enable "Voice Input" or "Voice Output"
3. Grant microphone permissions when prompted

### Using Voice Input

1. Click the microphone button or use the hotkey
2. Speak your question clearly
3. The system will convert speech to text
4. Review and submit your question

### Voice Output

- Enable "Voice Output" to have results read aloud
- Useful for accessibility or hands-free operation
- Adjust speaking rate in settings

## Tips and Best Practices

### Writing Effective Questions

#### ✅ Good Questions
```
"What's the average bill length for each penguin species?"
"Show me the correlation between age and survival in the titanic data"
"How many patients have missing blood pressure readings?"
```

#### ❌ Avoid These
```
"Show me everything" (too broad)
"What's good?" (unclear)
"Fix the data" (not specific)
```

### Getting Better Results

1. **Be Specific**: Include table names and column names when known
2. **Use Context**: Reference previous questions for follow-ups
3. **Ask One Thing**: Break complex questions into steps
4. **Use Examples**: "Like the penguins analysis, but for titanic data"

### Working with Large Datasets

1. **Start Small**: Use `LIMIT` in questions: "Show me the first 100 records"
2. **Filter First**: "Show me data for Adelie penguins only"
3. **Use Aggregations**: Ask for summaries rather than raw data
4. **Sample Data**: "Give me a random sample of 1000 records"

### Magic Command Tips

1. **Start with `/help`** to see all available commands
2. **Use tab completion** (if available) for command names
3. **Chain commands**: Use results from one command as input to another
4. **Save favorites**: Bookmark frequently used command patterns

## Troubleshooting

### Common Issues and Solutions

#### "No results found"
- Check table and column names
- Verify data exists with: "How many records are in [table]?"
- Try broader questions first

#### "Query error"
- Simplify your question
- Check for typos in table/column names
- Try asking about table structure first

#### "Connection error"
- Check with your administrator
- Verify database is running
- Try refreshing the page

#### Slow responses
- Use more specific questions
- Filter large datasets
- Ask for summaries instead of raw data

### Getting Help

1. **Use `/help`** for command assistance
2. **Check recent messages** for similar examples
3. **Start with simple questions** to verify connectivity
4. **Contact your administrator** for technical issues

### Performance Tips

- **Limit result sets**: Use "top 10" or "first 100" in questions
- **Filter early**: Specify conditions upfront
- **Use indexes**: Ask about commonly queried columns
- **Cache results**: Similar questions will be faster on repeat

### Data Quality Issues

If you encounter data quality problems:

1. **Document issues**: Note what you found
2. **Use cleaning commands**: Try `/missing` and `/duplicates`
3. **Report to admin**: Share findings with data stewards
4. **Work around**: Filter out problematic data for analysis

---

## Quick Reference Card

### Essential Commands
| Command | Purpose | Example |
|---------|---------|---------|
| `/describe table` | Statistical summary | `/describe penguins` |
| `/profile table` | Data quality report | `/profile titanic_train` |
| `/help` | Show all commands | `/help` |
| `question?` | Natural language query | `"What's the average age?"` |

### Common Questions
- "How many records are in [table]?"
- "What columns are in [table]?"
- "Show me the first 10 rows of [table]"
- "What's the average [column] by [group]?"

### Need Help?
- Type `/help` for command list
- Contact your administrator for technical support
- Check the troubleshooting section above

---

*Happy analyzing with Thrive AI! 🚀📊*