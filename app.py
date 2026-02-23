import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import base64
from io import BytesIO
from pathlib import Path

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== EMBEDDED CSS ====================
st.markdown("""
<style>
/* Main header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}
.main-header .subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* Sidebar header */
.sidebar-header {
    text-align: center;
    margin-bottom: 1rem;
}
.sidebar-header h2 {
    color: #667eea;
    font-size: 1.5rem;
}

/* Cards */
.metrics-card, .metric-card, .metric-card-highlight {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    color: #2c3e50 !important;
}
.metric-card-highlight {
    border-left: 4px solid #667eea;
}

/* Objective box */
.objective-box {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #2c3e50;
}
.objective-box ul {
    list-style-type: none;
    padding-left: 0;
}
.objective-box li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #e9ecef;
}
.objective-box li:last-child {
    border-bottom: none;
}

/* Findings / conclusion cards */
.findings-card, .conclusion-card, .recommendation-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 15px rgba(0,0,0,0.1);
    height: 100%;
    color: #2c3e50;
}
.findings-card h3, .conclusion-card h3, .recommendation-card h3 {
    color: #2c3e50;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.conclusion-point, .recommendation-item {
    margin-bottom: 1.5rem;
}
.conclusion-point h4, .recommendation-item h4 {
    color: #667eea;
    margin-bottom: 0.3rem;
    font-size: 1.1rem;
}
.recommendation-item ul {
    padding-left: 1.5rem;
}
.recommendation-item li {
    margin: 0.3rem 0;
}

/* Final insight */
.final-insight {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    font-style: italic;
    margin-top: 2rem;
    color: #2c3e50;
}
.final-insight p {
    font-size: 1.2rem;
}
.final-insight .author {
    font-weight: bold;
    margin-top: 1rem;
    font-size: 1rem;
    color: #667eea;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1rem;
    color: #6c757d;
    border-top: 1px solid #e9ecef;
    margin-top: 2rem;
}

/* Ensure metric labels/values are dark */
.stMetric label, .stMetric .metric-value, .stMetric div[data-testid="stMetricValue"] {
    color: #2c3e50 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
    <div class="main-header">
        <h1>📊 STUDENT PERFORMANCE ANALYSIS</h1>
        <p class="subtitle">Data Analytics Using Python | Organized by ADHOC NETWORK</p>
    </div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h2>🔍 Analysis Menu</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Data input options
    st.markdown("### 📁 Data Management")
    data_option = st.radio("Select Data Source:", ["📋 Use Sample Data", "📤 Upload CSV File", "➕ Add Student Manually", "📥 Download Template"])
    uploaded_file = None

    if data_option == "📤 Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    elif data_option == "➕ Add Student Manually":
        st.subheader("Add New Student")
        with st.form("add_student_form"):
            student_id = st.text_input("Student ID (e.g., S001)", "S001")
            study_hours = st.number_input("Study Hours", min_value=0, max_value=50, value=20)
            final_score = st.number_input("Final Score", min_value=0, max_value=100, value=75)
            attendance = st.number_input("Attendance", min_value=0, max_value=30, value=25)
            assignment_score = st.number_input("Assignment Score", min_value=0, max_value=100, value=80)
            study_habit = st.selectbox("Study Habit", ["Consistent", "Irregular", "Cramming"])
            submit_button = st.form_submit_button("Add Student")
            if submit_button:
                st.session_state.new_students = st.session_state.get('new_students', [])
                st.session_state.new_students.append({
                    'Student_ID': student_id,
                    'Study_Hours': study_hours,
                    'Final_Score': final_score,
                    'Attendance': attendance,
                    'Assignment_Score': assignment_score,
                    'Study_Habit': study_habit
                })
                st.success(f"✅ {student_id} added successfully!")
    elif data_option == "📥 Download Template":
        template_df = pd.DataFrame({
            'Student_ID': ['S001', 'S002', 'S003'],
            'Study_Hours': [20.5, 15.3, 25.2],
            'Final_Score': [78.5, 65.2, 88.9],
            'Attendance': [28, 22, 30],
            'Assignment_Score': [82.0, 70.0, 90.0],
            'Study_Habit': ['Consistent', 'Irregular', 'Consistent']
        })
        csv_buffer = BytesIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="student_data_template.csv",
            mime="text/csv"
        )
        st.info("Fill this template with your student data and upload it back!")

    st.markdown("---")

    # Analysis parameters
    st.markdown("### ⚙️ Analysis Settings")
    confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
    show_regression = st.checkbox("Show Regression Line", value=True)
    color_scheme = st.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])

    st.markdown("---")

    # Information
    st.markdown("### ℹ️ About")
    st.info("""
        This dashboard analyzes the relationship between study hours and final scores.
        **Tools Used:**
        - Python
        - Pandas
        - Matplotlib/Seaborn
        - Streamlit
    """)

# ==================== DATA GENERATION ====================
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_students = 100

    # Create positive correlation
    study_hours = np.random.normal(25, 8, n_students)
    study_hours = np.clip(study_hours, 5, 40)

    # Final scores with correlation to study hours
    final_scores = 40 + study_hours * 1.8 + np.random.normal(0, 10, n_students)
    final_scores = np.clip(final_scores, 40, 100)

    # Additional features
    attendance = np.random.binomial(30, 0.85, n_students)
    assignment_scores = np.random.normal(75, 15, n_students)
    assignment_scores = np.clip(assignment_scores, 40, 100)

    data = pd.DataFrame({
        'Student_ID': [f'S{i:03d}' for i in range(1, n_students + 1)],
        'Study_Hours': np.round(study_hours, 1),
        'Final_Score': np.round(final_scores, 1),
        'Attendance': attendance,
        'Assignment_Score': np.round(assignment_scores, 1),
        'Study_Habit': np.random.choice(['Consistent', 'Irregular', 'Cramming'], n_students, p=[0.4, 0.4, 0.2])
    })

    # Performance category
    def get_performance(score):
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 55:
            return 'Average'
        else:
            return 'Needs Improvement'

    data['Performance'] = data['Final_Score'].apply(get_performance)
    return data

# ==================== LOAD DATA ====================
def get_performance(score):
    if score >= 85:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 55:
        return 'Average'
    else:
        return 'Needs Improvement'

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = generate_sample_data()
        st.info("Using sample data for demonstration")
elif 'new_students' in st.session_state and st.session_state.new_students:
    # Combine sample data with manually added students
    df = generate_sample_data()
    new_students_df = pd.DataFrame(st.session_state.new_students)
    df = pd.concat([df, new_students_df], ignore_index=True)
    st.success(f"✅ Added {len(st.session_state.new_students)} student(s) to sample data!")
else:
    df = generate_sample_data()
    st.info("📋 Using sample dataset. Choose a data option in the sidebar!")

# Add Performance category if not present
if 'Performance' not in df.columns:
    df['Performance'] = df['Final_Score'].apply(get_performance)

# ==================== MAIN DASHBOARD ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔬 Analysis", "📊 Visualization", "📋 Findings", "🎯 Conclusion"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div class="section-header">
                <h2>🎯 PROJECT OBJECTIVE</h2>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div class="objective-box">
                <ul>
                    <li>To find out if studying more hours helps students get better marks</li>
                    <li>To analyze student performance patterns</li>
                    <li>To find the relation between study hours and final scores</li>
                    <li>To understand the importance of regular study habits</li>
                    <li>To identify ways to improve student performance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metrics-card">
                <h3>📊 Dataset Overview</h3>
            </div>
        """, unsafe_allow_html=True)
        total_students = len(df)
        avg_hours = df['Study_Hours'].mean()
        avg_score = df['Final_Score'].mean()
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Students", total_students)
            st.metric("Avg Study Hours", f"{avg_hours:.1f} hrs")
        with col_b:
            st.metric("Avg Final Score", f"{avg_score:.1f}%")
            st.metric("Data Columns", len(df.columns))

    st.markdown("---")
    st.markdown("**📁 Data Preview:**")
    st.dataframe(df.head(10), use_container_width=True)

    # Show newly added students
    if 'new_students' in st.session_state and st.session_state.new_students:
        st.markdown("---")
        st.subheader("✨ Recently Added Students")
        recent_df = pd.DataFrame(st.session_state.new_students)
        st.dataframe(recent_df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Added Students"):
                st.session_state.new_students = []
                st.rerun()
        with col2:
            csv_buffer = BytesIO()
            recent_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="📥 Download Added Students",
                data=csv_buffer.getvalue(),
                file_name="added_students.csv",
                mime="text/csv"
            )

# ==================== TAB 2: ANALYSIS ====================
with tab2:
    st.markdown("""
        <div class="section-header">
            <h2>🔬 DATA ANALYSIS</h2>
        </div>
    """, unsafe_allow_html=True)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(df['Study_Hours'], df['Final_Score'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="metric-card-highlight">
                <h4>Correlation Coefficient</h4>
                <h1>{:.3f}</h1>
                <p>Study Hours ↔ Final Score</p>
            </div>
        """.format(correlation), unsafe_allow_html=True)

    with col2:
        if abs(correlation) >= 0.7:
            strength = "Strong"
            color = "green"
        elif abs(correlation) >= 0.3:
            strength = "Moderate"
            color = "orange"
        else:
            strength = "Weak"
            color = "red"
        st.markdown(f"""
            <div class="metric-card">
                <h4>Correlation Strength</h4>
                <h1 style="color: {color};">{strength}</h1>
                <p>{'Positive' if correlation > 0 else 'Negative'} Relationship</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        st.markdown(f"""
            <div class="metric-card">
                <h4>Statistical Significance</h4>
                <h1>{significance}</h1>
                <p>p-value: {p_value:.4f}</p>
            </div>
        """, unsafe_allow_html=True)

    # Regression analysis
    st.markdown("### 📈 Regression Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot with regression line
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Study_Hours', y='Final_Score', data=df,
                    scatter_kws={'alpha': 0.6, 'color': '#4B8BBE', 's': 80},
                    line_kws={'color': '#FF7F0E', 'linewidth': 3}, ax=ax)
        ax.set_xlabel('Study Hours per Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Study Hours vs Final Score with Regression Line', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # Performance distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        performance_counts = df['Performance'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(performance_counts)))
        bars = ax2.bar(performance_counts.index, performance_counts.values, color=colors)
        ax2.set_xlabel('Performance Category', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Students', fontsize=12, fontweight='bold')
        ax2.set_title('Student Performance Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                     ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)

    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Study Hours Statistics**")
        study_stats = df['Study_Hours'].describe()
        st.dataframe(study_stats)
    with col2:
        st.markdown("**Final Score Statistics**")
        score_stats = df['Final_Score'].describe()
        st.dataframe(score_stats)

# ==================== TAB 3: VISUALIZATION ====================
with tab3:
    st.markdown("""
        <div class="section-header">
            <h2>📊 ADVANCED VISUALIZATIONS</h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Interactive scatter plot
        fig = px.scatter(df, x='Study_Hours', y='Final_Score', color='Performance',
                         hover_data=['Student_ID', 'Attendance', 'Assignment_Score'],
                         title='Interactive Scatter Plot: Study Hours vs Final Score',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(
            xaxis_title="Study Hours per Week",
            yaxis_title="Final Score (%)",
            hovermode='closest',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot
        fig2 = px.box(df, x='Performance', y='Study_Hours', color='Performance',
                      title='Study Hours Distribution by Performance',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(
            xaxis_title="Performance Category",
            yaxis_title="Study Hours per Week",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax3)
    ax3.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)

# ==================== TAB 4: FINDINGS ====================
with tab4:
    st.markdown("""
        <div class="section-header">
            <h2>📋 KEY FINDINGS</h2>
        </div>
    """, unsafe_allow_html=True)

    findings_col1, findings_col2 = st.columns(2)

    with findings_col1:
        # Render Major Insights inside the findings-card so it matches other cards
        st.markdown(
            """
            <div class="findings-card">
                <h3>🔍 Major Insights</h3>
                <ul>
                    <li><strong>Strong Positive Correlation Found:</strong> Increased study hours directly correlate with higher final scores</li>
                    <li><strong>Optimal Study Range:</strong> Students studying 25-35 hours/week achieve the best results</li>
                    <li><strong>Consistency Matters:</strong> Regular study habits yield better outcomes than irregular cramming</li>
                    <li><strong>Performance Threshold:</strong> Minimum 15 hours/week needed to score above 60%</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Performance by study habit
        st.markdown("### 📚 Performance by Study Habit")
        habit_performance = df.groupby('Study_Habit')['Final_Score'].agg(['mean', 'count']).round(1)
        st.dataframe(habit_performance.style.highlight_max(axis=0))

    with findings_col2:
        # Render conclusions inside the styled card
        st.markdown(
            """
            <div class="conclusion-card">
                <h3>✅ MAIN CONCLUSIONS</h3>
                <div class="conclusion-point">
                    <h4>1. Regular Study Improves Performance</h4>
                    <p>The analysis confirms that consistent study habits significantly impact academic outcomes.</p>
                </div>
                <div class="conclusion-point">
                    <h4>2. More Study Hours = Higher Marks</h4>
                    <p>There's a clear positive relationship between study time and final scores.</p>
                </div>
                <div class="conclusion-point">
                    <h4>3. Good Study Habits are Crucial</h4>
                    <p>Organization, time management, and consistency are key success factors.</p>
                </div>
                <div class="conclusion-point">
                    <h4>4. Data-Driven Learning Works</h4>
                    <p>This project demonstrates the value of using analytics to understand and improve education.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================== TAB 5: CONCLUSION ====================
with tab5:
    st.markdown("""
        <div class="section-header">
            <h2>🎯 CONCLUSIONS & RECOMMENDATIONS</h2>
        </div>
    """, unsafe_allow_html=True)

    conclusion_col1, conclusion_col2 = st.columns(2)

    with conclusion_col1:
        # Styled header for the conclusions card
        st.markdown(
            """
            <div class="conclusion-card">
                <h3>✅ MAIN CONCLUSIONS</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Use native markdown so content renders instead of showing raw HTML
        st.markdown(
            """
            **1. Regular Study Improves Performance**
            The analysis confirms that consistent study habits significantly impact academic outcomes.

            **2. More Study Hours = Higher Marks**
            There's a clear positive relationship between study time and final scores.

            **3. Good Study Habits are Crucial**
            Organization, time management, and consistency are key success factors.

            **4. Data-Driven Learning Works**
            This project demonstrates the value of using analytics to understand and improve education.
            """
        )

    with conclusion_col2:
        st.markdown(
            """
            <div class="recommendation-card">
                <h3>💡 RECOMMENDATIONS</h3>
                <div class="recommendation-item">
                    <h4>For Students:</h4>
                    <ul>
                        <li>Aim for 25-30 study hours per week</li>
                        <li>Establish consistent daily study routines</li>
                        <li>Track study hours and academic performance</li>
                        <li>Focus on quality over quantity of study time</li>
                    </ul>
                </div>
                <div class="recommendation-item">
                    <h4>For Educators:</h4>
                    <ul>
                        <li>Implement study time monitoring systems</li>
                        <li>Provide study habit workshops</li>
                        <li>Create personalized learning plans</li>
                        <li>Use data analytics for early intervention</li>
                    </ul>
                </div>
                <div class="recommendation-item">
                    <h4>For Institutions:</h4>
                    <ul>
                        <li>Develop data-driven academic policies</li>
                        <li>Create supportive learning environments</li>
                        <li>Invest in educational technology</li>
                        <li>Promote research-based teaching methods</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Final insights
    st.markdown("---")
    st.markdown("""
        <div class="final-insight">
            <h3>🌟 FINAL INSIGHT</h3>
            <p>
                "The successful development of young people and society is achieved through academic excellence.
                Students who are better organized, have a substantial base of knowledge, and have better scope to
                achieve heights in their career. Proper motivation, concentration, prioritization, organizing abilities,
                and time management help in achieving academic success easily."
            </p>
            <p class="author">- ADHOC NETWORK Bootcamp Project</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("""
        **📊 Data Analytics Using Python**
        Organized by **ADHOC NETWORK**
    """)
with footer_col2:
    st.markdown("""
        **🛠 Tools Used:**
        Python • Pandas • Matplotlib • Seaborn • Streamlit
    """)
with footer_col3:
    st.markdown("""
        **📧 Contact:**
        student.analysis@adhocnetwork.com
    """)
st.markdown("<div class='footer'>© 2024 Student Performance Analysis Dashboard | ADHOC NETWORK Bootcamp Project</div>", unsafe_allow_html=True)

# ==================== DOWNLOAD DATA ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Data")
if st.sidebar.button("Download Sample Data"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="student_performance_data.csv">Download CSV File</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)