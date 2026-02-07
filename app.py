from dotenv import load_dotenv
import os

load_dotenv()
"""
Generative AI Powered Curriculum Design System
A comprehensive system for designing academic curricula using AI
"""

import os
import json
from dotenv import load_dotenv
import gradio as gr
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


# ============================================================================
# CORE CURRICULUM GENERATION FUNCTIONS
# ============================================================================

def generate_course_structure(subject, level, duration, goals):
    """
    Generate a structured course outline with modules and topics.
    
    Args:
        subject: The subject/course name
        level: Academic level (school/college/university)
        duration: Duration in weeks or semesters
        goals: Learning goals and objectives
    
    Returns:
        Structured course outline as formatted text
    """
    prompt = f"""You are an expert academic curriculum designer. Generate a comprehensive course structure for:

Subject: {subject}
Academic Level: {level}
Duration: {duration}
Learning Goals: {goals}

Create a detailed course structure with:
1. Course title and description
2. 5-8 major modules/units
3. 3-5 topics under each module
4. Brief description for each module

Format the response clearly with headers and bullet points. Be structured and academic."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic curriculum designer who creates structured, comprehensive course outlines."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating course structure: {str(e)}"


def recommend_topics(subject, level, goals, focus_area):
    """
    Suggest relevant topics based on subject, level, and goals.
    
    Args:
        subject: The subject/course name
        level: Academic level
        goals: Learning goals
        focus_area: Specific area of focus (optional)
    
    Returns:
        List of recommended topics with justifications
    """
    prompt = f"""You are an expert academic curriculum designer. Recommend relevant topics for:

Subject: {subject}
Academic Level: {level}
Learning Goals: {goals}
Focus Area: {focus_area if focus_area else "General coverage"}

Provide:
1. 10-15 recommended topics
2. Brief justification for each topic's inclusion
3. Suggested depth of coverage (introductory/intermediate/advanced)
4. Prerequisites if any

Be specific and academically rigorous."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic curriculum designer who recommends relevant and impactful learning topics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error recommending topics: {str(e)}"


def create_curriculum_plan(subject, level, duration, goals, structure):
    """
    Create a semester-wise or timeline-based curriculum plan.
    
    Args:
        subject: The subject/course name
        level: Academic level
        duration: Duration (e.g., "1 semester", "12 weeks")
        goals: Learning goals
        structure: Previously generated course structure (optional)
    
    Returns:
        Timeline-based curriculum plan
    """
    prompt = f"""You are an expert academic curriculum designer. Create a detailed timeline-based curriculum plan for:

Subject: {subject}
Academic Level: {level}
Duration: {duration}
Learning Goals: {goals}

Create a week-by-week or session-by-session plan that includes:
1. Clear timeline (Week 1, Week 2, etc. or Session 1, Session 2, etc.)
2. Topics to be covered in each period
3. Suggested activities (lectures, labs, assignments, assessments)
4. Estimated hours per topic
5. Key milestones and assessment points

Ensure logical progression from basics to advanced concepts. Be realistic about pacing."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic curriculum designer who creates realistic, well-paced learning schedules."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error creating curriculum plan: {str(e)}"


def map_learning_outcomes(subject, level, modules_topics):
    """
    Map each topic/module to measurable learning outcomes.
    
    Args:
        subject: The subject/course name
        level: Academic level
        modules_topics: Description of modules and topics
    
    Returns:
        Learning outcomes mapped to modules/topics
    """
    prompt = f"""You are an expert academic curriculum designer. Create measurable learning outcomes for:

Subject: {subject}
Academic Level: {level}
Content: {modules_topics}

For each major module/topic, define:
1. 3-5 specific learning outcomes using Bloom's Taxonomy
2. Use action verbs (understand, analyze, create, evaluate, apply, etc.)
3. Make outcomes measurable and assessable
4. Align outcomes with the academic level
5. Include cognitive, skill-based, and affective outcomes where appropriate

Format: Module/Topic → Learning Outcomes (numbered list)

Be precise and use proper educational outcome language."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic curriculum designer who writes clear, measurable learning outcomes aligned with Bloom's Taxonomy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error mapping learning outcomes: {str(e)}"


def optimize_curriculum(subject, level, duration, goals, current_plan):
    """
    Optimize curriculum for difficulty progression, relevance, and balance.
    
    Args:
        subject: The subject/course name
        level: Academic level
        duration: Duration
        goals: Learning goals
        current_plan: Current curriculum plan to optimize
    
    Returns:
        Optimized curriculum with recommendations
    """
    prompt = f"""You are an expert academic curriculum designer. Optimize the curriculum for:

Subject: {subject}
Academic Level: {level}
Duration: {duration}
Learning Goals: {goals}
Current Plan Summary: {current_plan[:500] if current_plan else "Generate fresh optimization"}

Analyze and optimize for:
1. **Difficulty Progression**: Ensure smooth transition from basic to advanced
2. **Topic Relevance**: Prioritize high-impact, industry-relevant topics
3. **Academic Balance**: Balance theory, practice, and assessments
4. **Cognitive Load**: Avoid overwhelming students in any period
5. **Redundancy**: Identify and eliminate duplicate content
6. **Gaps**: Identify missing critical topics

Provide:
- Overall curriculum health score (0-100)
- Specific optimization recommendations
- Suggested reordering or restructuring
- Balance metrics (theory vs practice ratio, assessment distribution)

Be critical and constructive."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert academic curriculum designer who optimizes learning experiences for maximum effectiveness and balance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error optimizing curriculum: {str(e)}"


# ============================================================================
# GRADIO UI FUNCTIONS
# ============================================================================

def generate_full_curriculum(subject, level, duration, goals, focus_area):
    """
    Generate complete curriculum with all components.
    """
    if not subject or not level or not duration or not goals:
        return "❌ Please fill in all required fields: Subject, Level, Duration, and Goals."
    
    output = "# 🎓 COMPREHENSIVE CURRICULUM DESIGN SYSTEM\n\n"
    output += f"**Subject:** {subject} | **Level:** {level} | **Duration:** {duration}\n\n"
    output += "---\n\n"
    
    # Generate each component
    output += "## 📚 1. COURSE STRUCTURE\n\n"
    structure = generate_course_structure(subject, level, duration, goals)
    output += structure + "\n\n---\n\n"
    
    output += "## 💡 2. RECOMMENDED TOPICS\n\n"
    topics = recommend_topics(subject, level, goals, focus_area)
    output += topics + "\n\n---\n\n"
    
    output += "## 📅 3. CURRICULUM TIMELINE\n\n"
    timeline = create_curriculum_plan(subject, level, duration, goals, structure)
    output += timeline + "\n\n---\n\n"
    
    output += "## 🎯 4. LEARNING OUTCOMES MAPPING\n\n"
    outcomes = map_learning_outcomes(subject, level, structure)
    output += outcomes + "\n\n---\n\n"
    
    output += "## ⚡ 5. CURRICULUM OPTIMIZATION\n\n"
    optimization = optimize_curriculum(subject, level, duration, goals, timeline)
    output += optimization + "\n\n---\n\n"
    
    output += "\n\n✅ **Curriculum design complete!** Review each section above for your comprehensive academic plan."
    
    return output


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """
    Create the Gradio interface for the curriculum design system.
    """
    
    # Custom CSS for professional styling
    css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .output-box {
        font-size: 14px;
        line-height: 1.6;
    }
    """
    
    with gr.Blocks(css=css, title="AI Curriculum Designer") as interface:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🎓 AI-Powered Curriculum Design System</h1>
                <p>Comprehensive curriculum generation for educational excellence</p>
            </div>
        """)
        
        gr.Markdown("""
        ### Welcome to the Curriculum Design System
        This AI-powered tool helps you create comprehensive, optimized academic curricula. 
        Fill in the details below to generate a complete curriculum with structure, timeline, and learning outcomes.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Curriculum Inputs")
                
                subject_input = gr.Textbox(
                    label="Subject/Course Name *",
                    placeholder="e.g., Data Structures and Algorithms",
                    lines=1
                )
                
                level_input = gr.Dropdown(
                    label="Academic Level *",
                    choices=[
                        "High School",
                        "Undergraduate (Bachelor's)",
                        "Graduate (Master's)",
                        "Doctoral (PhD)",
                        "Professional Certification",
                        "Continuing Education"
                    ],
                    value="Undergraduate (Bachelor's)"
                )
                
                duration_input = gr.Textbox(
                    label="Duration *",
                    placeholder="e.g., 1 semester, 12 weeks, 6 months",
                    lines=1
                )
                
                goals_input = gr.TextArea(
                    label="Learning Goals & Objectives *",
                    placeholder="Describe what students should achieve by the end of the course...",
                    lines=4
                )
                
                focus_input = gr.Textbox(
                    label="Focus Area (Optional)",
                    placeholder="e.g., Industry applications, Research focus, Practical skills",
                    lines=1
                )
                
                generate_btn = gr.Button("🚀 Generate Complete Curriculum", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                **Individual Components:**  
                Use buttons below to generate specific sections:
                """)
                
                with gr.Row():
                    structure_btn = gr.Button("📚 Course Structure", size="sm")
                    topics_btn = gr.Button("💡 Topic Recommendations", size="sm")
                
                with gr.Row():
                    timeline_btn = gr.Button("📅 Timeline Plan", size="sm")
                    outcomes_btn = gr.Button("🎯 Learning Outcomes", size="sm")
                
                optimize_btn = gr.Button("⚡ Optimize Curriculum", size="sm")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Generated Curriculum")
                
                output_display = gr.Markdown(
                    value="Your generated curriculum will appear here...",
                    elem_classes="output-box"
                )
        
        # Event handlers for full curriculum generation
        generate_btn.click(
            fn=generate_full_curriculum,
            inputs=[subject_input, level_input, duration_input, goals_input, focus_input],
            outputs=output_display
        )
        
        # Event handlers for individual components
        structure_btn.click(
            fn=lambda s, l, d, g: "## 📚 COURSE STRUCTURE\n\n" + generate_course_structure(s, l, d, g),
            inputs=[subject_input, level_input, duration_input, goals_input],
            outputs=output_display
        )
        
        topics_btn.click(
            fn=lambda s, l, g, f: "## 💡 RECOMMENDED TOPICS\n\n" + recommend_topics(s, l, g, f),
            inputs=[subject_input, level_input, goals_input, focus_input],
            outputs=output_display
        )
        
        timeline_btn.click(
            fn=lambda s, l, d, g: "## 📅 CURRICULUM TIMELINE\n\n" + create_curriculum_plan(s, l, d, g, ""),
            inputs=[subject_input, level_input, duration_input, goals_input],
            outputs=output_display
        )
        
        outcomes_btn.click(
            fn=lambda s, l: "## 🎯 LEARNING OUTCOMES\n\n" + map_learning_outcomes(s, l, "Course content as defined"),
            inputs=[subject_input, level_input],
            outputs=output_display
        )
        
        optimize_btn.click(
            fn=lambda s, l, d, g: "## ⚡ CURRICULUM OPTIMIZATION\n\n" + optimize_curriculum(s, l, d, g, ""),
            inputs=[subject_input, level_input, duration_input, goals_input],
            outputs=output_display
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📖 How to Use:
        1. **Fill in required fields** (marked with *)
        2. **Click "Generate Complete Curriculum"** for a full curriculum design
        3. **Or use individual buttons** to generate specific sections
        4. **Review and refine** the generated content for your needs
        
        **Powered by:** Groq AI (Llama 3.3 70B) | **Built for:** College Hackathon Project
        """)
    
    return interface


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in .env file")
        print("Please create a .env file with: GROQ_API_KEY=your_api_key_here")
        exit(1)
    
    print("🚀 Starting AI Curriculum Design System...")
    print("📍 Access the application at: http://localhost:7860")
    
    # Create and launch the interface
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
    
ui.launch()

