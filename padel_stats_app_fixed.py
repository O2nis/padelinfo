import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from collections import defaultdict

def parse_date_robust(date_str):
    """
    Robustly parse date from various formats including Excel serial numbers.
    Tries common formats: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, Excel serial, etc.
    """
    if pd.isna(date_str):
        return None
    
    # If already datetime, return as is
    if isinstance(date_str, datetime):
        return date_str
    
    # Check if it's a number (Excel serial date)
    try:
        num_value = float(date_str)
        if num_value > 1000:  # Likely an Excel serial number
            # Excel epoch starts at 1899-12-30
            return pd.to_datetime('1899-12-30') + pd.to_timedelta(num_value, unit='D')
    except (ValueError, TypeError):
        pass
    
    date_str = str(date_str).strip()
    
    # Try various date formats
    date_formats = [
        "%d/%m/%Y",  # 10/02/2025
        "%m/%d/%Y",  # 02/10/2025
        "%Y-%m-%d",  # 2025-02-10
        "%Y/%m/%d",  # 2025/02/10
        "%d-%m-%Y",  # 10-02-2025
        "%d.%m.%Y",  # 10.02.2025
        "%A, %B %d, %Y",  # Thursday, October 2, 2025
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all fail, try pandas to_datetime
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return None

def clean_score(score_value):
    """
    Clean and convert score to float, handling various formats.
    """
    if pd.isna(score_value):
        return None
    
    # Convert to string and strip whitespace
    score_str = str(score_value).strip()
    
    # Try to convert to float
    try:
        return float(score_str)
    except (ValueError, TypeError):
        return None

def load_and_process_data(df):
    """
    Process the padel match data and calculate statistics.
    """
    # Parse dates
    df['Date_parsed'] = df['Date'].apply(parse_date_robust)
    df['Date_only'] = df['Date_parsed'].dt.date
    
    # Initialize player statistics
    player_stats = defaultdict(lambda: {
        'wins': 0, 'losses': 0, 'draws': 0,
        'total_points': 0, 'bonus_points': 0,
        'dates_played': set(),
        'wins_by_date': defaultdict(int)
    })
    
    all_dates = sorted(df['Date_only'].dropna().unique())
    
    # DEBUG: Track draw detection
    draws_detected = []
    total_matches = 0
    
    # Process each match
    for idx, row in df.iterrows():
        if pd.isna(row['Date_parsed']):
            continue
            
        total_matches += 1
        date = row['Date_only']
        
        # Clean and convert scores to float
        score1 = clean_score(row['Score1'])
        score2 = clean_score(row['Score2'])
        
        # Skip if scores are invalid
        if score1 is None or score2 is None:
            continue
        
        # Get players
        team1 = [row['Player1'], row['Player2']]
        team2 = [row['Player3'], row['Player4']]
        
        # DEBUG: Print score comparison for first few matches
        if total_matches <= 5:
            print(f"Match {total_matches}: {score1} vs {score2} -> Equal: {score1 == score2}")
        
        # Determine match result with EXPLICIT draw detection
        is_draw = (score1 == score2)
        team1_wins = (score1 > score2)
        team2_wins = (score2 > score1)
        
        # DEBUG: Track draws
        if is_draw:
            draws_detected.append({
                'Match_Number': total_matches,
                'Date': date,
                'Team1': f"{team1[0]} & {team1[1]}",
                'Team2': f"{team2[0]} & {team2[1]}",
                'Score1': score1,
                'Score2': score2,
                'Equal': score1 == score2
            })
            print(f"DRAW DETECTED: Match {total_matches} - {team1[0]} & {team1[1]} vs {team2[0]} & {team2[1]} - Score: {score1}-{score2}")
        
        # Assign results
        if is_draw:
            result_team1 = 'draw'
            result_team2 = 'draw'
        elif team1_wins:
            result_team1 = 'win'
            result_team2 = 'loss'
        elif team2_wins:
            result_team1 = 'loss'
            result_team2 = 'win'
        else:
            # This should never happen, but just in case
            result_team1 = 'draw'
            result_team2 = 'draw'
        
        # Update statistics for Team1
        for player in team1:
            if pd.notna(player):
                player_stats[player]['dates_played'].add(date)
                if result_team1 == 'win':
                    player_stats[player]['wins'] += 1
                    player_stats[player]['wins_by_date'][date] += 1
                    player_stats[player]['total_points'] += 3
                elif result_team1 == 'draw':
                    player_stats[player]['draws'] += 1
                    player_stats[player]['total_points'] += 1
                else:  # loss
                    player_stats[player]['losses'] += 1
        
        # Update statistics for Team2
        for player in team2:
            if pd.notna(player):
                player_stats[player]['dates_played'].add(date)
                if result_team2 == 'win':
                    player_stats[player]['wins'] += 1
                    player_stats[player]['wins_by_date'][date] += 1
                    player_stats[player]['total_points'] += 3
                elif result_team2 == 'draw':
                    player_stats[player]['draws'] += 1
                    player_stats[player]['total_points'] += 1
                else:  # loss
                    player_stats[player]['losses'] += 1
    
    # Calculate bonus points
    for player in player_stats:
        for date, wins_count in player_stats[player]['wins_by_date'].items():
            if wins_count >= 4:
                player_stats[player]['bonus_points'] += 2
            elif wins_count == 3:
                player_stats[player]['bonus_points'] += 1
    
    print(f"Total matches processed: {total_matches}")
    print(f"Draws detected: {len(draws_detected)}")
    
    return player_stats, all_dates, draws_detected

def create_chart1_wld_stacked(player_stats):
    """
    Chart 1: Vertical stacked bar chart showing wins, losses, draws per player (sorted by wins).
    """
    # Sort players by wins (descending)
    sorted_players = sorted(player_stats.keys(), 
                           key=lambda p: player_stats[p]['wins'], 
                           reverse=True)
    
    wins = [player_stats[p]['wins'] for p in sorted_players]
    draws = [player_stats[p]['draws'] for p in sorted_players]
    losses = [player_stats[p]['losses'] for p in sorted_players]
    
    fig = go.Figure()
    
    # Add traces for stacked bar chart
    fig.add_trace(go.Bar(
        name='Wins',
        x=sorted_players,
        y=wins,
        marker_color='#2ecc71',
        text=[f'{w}' if w > 0 else '' for w in wins],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Wins: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Draws',
        x=sorted_players,
        y=draws,
        marker_color='#f39c12',
        text=[f'{d}' if d > 0 else '' for d in draws],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Draws: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Losses',
        x=sorted_players,
        y=losses,
        marker_color='#e74c3c',
        text=[f'{l}' if l > 0 else '' for l in losses],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{x}</b><br>Losses: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Wins, Draws, and Losses per Player (Sorted by Wins)',
        xaxis_title='Players',
        yaxis_title='Number of Matches',
        font=dict(size=12),
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_chart2_points_stacked(player_stats):
    """
    Chart 2: Horizontal stacked bar chart showing total points and bonus points (sorted by total points, HIGHEST ON TOP).
    """
    # Sort players by total points + bonus (descending)
    sorted_players = sorted(player_stats.keys(), 
                           key=lambda p: player_stats[p]['total_points'] + player_stats[p]['bonus_points'], 
                           reverse=True)
    
    total_points = [player_stats[p]['total_points'] for p in sorted_players]
    bonus_points = [player_stats[p]['bonus_points'] for p in sorted_players]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Total Points',
        y=sorted_players,
        x=total_points,
        orientation='h',
        marker_color='#3498db',
        text=[f'{p}' if p > 0 else '' for p in total_points],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Total Points: %{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Bonus Points',
        y=sorted_players,
        x=bonus_points,
        orientation='h',
        marker_color='#9b59b6',
        text=[f'{p}' if p > 0 else '' for p in bonus_points],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Bonus Points: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Total Points and Bonus Points per Player (Sorted by Total - Highest on Top)',
        xaxis_title='Points',
        yaxis=dict(
            title='Players',
            autorange='reversed'  # This puts the first item (highest score) on TOP
        ),
        font=dict(size=12),
        height=max(400, len(sorted_players) * 40),
        hovermode='y unified'
    )
    
    return fig

def create_chart3_timeline_detailed(df, player_stats, all_dates):
    """
    Chart 3: Detailed timeline with W/D/L counts per date and connecting lines.
    """
    # Sort players by wins (descending)
    sorted_players = sorted(player_stats.keys(), 
                           key=lambda p: player_stats[p]['wins'], 
                           reverse=True)
    
    # Process data to get W/D/L per player per date
    player_date_stats = defaultdict(lambda: defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0}))
    
    df['Date_parsed'] = df['Date'].apply(parse_date_robust)
    df['Date_only'] = df['Date_parsed'].dt.date
    
    for idx, row in df.iterrows():
        if pd.isna(row['Date_parsed']):
            continue
            
        date = row['Date_only']
        
        # Clean and convert scores
        score1 = clean_score(row['Score1'])
        score2 = clean_score(row['Score2'])
        
        # Skip if scores are invalid
        if score1 is None or score2 is None:
            continue
        
        team1 = [row['Player1'], row['Player2']]
        team2 = [row['Player3'], row['Player4']]
        
        # Determine match result with EXPLICIT draw detection
        is_draw = (score1 == score2)
        team1_wins = (score1 > score2)
        
        if is_draw:
            result_team1, result_team2 = 'D', 'D'
        elif team1_wins:
            result_team1, result_team2 = 'W', 'L'
        else:
            result_team1, result_team2 = 'L', 'W'
        
        for player in team1:
            if pd.notna(player):
                player_date_stats[player][date][result_team1] += 1
        
        for player in team2:
            if pd.notna(player):
                player_date_stats[player][date][result_team2] += 1
    
    fig = go.Figure()
    
    # Create traces for each player
    for idx, player in enumerate(sorted_players):
        played_dates = []
        not_played_dates = []
        played_y = []
        not_played_y = []
        annotations = []
        
        for date in all_dates:
            stats = player_date_stats[player][date]
            played = stats['W'] + stats['D'] + stats['L'] > 0
            
            if played:
                played_dates.append(date)
                played_y.append(idx)
                annotations.append(f"W={stats['W']}, D={stats['D']}, L={stats['L']}")
            else:
                not_played_dates.append(date)
                not_played_y.append(idx)
        
        # Add connecting line for all dates (makes timeline continuous)
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=[idx] * len(all_dates),
            mode='lines',
            line=dict(color='lightgray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add green dots for played dates
        if played_dates:
            fig.add_trace(go.Scatter(
                x=played_dates,
                y=played_y,
                mode='markers+text',
                name=player,
                marker=dict(
                    size=15,
                    color='#2ecc71',
                    line=dict(color='black', width=2)
                ),
                text=annotations,
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>%{text}<extra></extra>'
            ))
        
        # Add red empty dots for not played dates
        if not_played_dates:
            fig.add_trace(go.Scatter(
                x=not_played_dates,
                y=not_played_y,
                mode='markers',
                name=f'{player} (not played)',
                marker=dict(
                    size=15,
                    color='white',
                    line=dict(color='#e74c3c', width=3)
                ),
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Did not play<extra></extra>'
            ))
    
    fig.update_layout(
        title='Player Participation Timeline (W=Wins, D=Draws, L=Losses)',
        xaxis_title='Date',
        yaxis=dict(
            title='Players',
            tickmode='array',
            tickvals=list(range(len(sorted_players))),
            ticktext=sorted_players
        ),
        font=dict(size=12),
        height=max(500, len(sorted_players) * 60),
        hovermode='closest',
        showlegend=False
    )
    
    return fig

# Streamlit App
def main():
    st.set_page_config(page_title="Padel Statistics Dashboard", layout="wide")
    
    st.title("🎾 Padel Statistics Dashboard")
    st.markdown("Upload your Excel file to analyze player statistics, points, and participation timeline.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_cols = ['Date', 'Court', 'Player1', 'Player2', 'Score1', 'Score2', 'Player3', 'Player4']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Excel must contain columns: {', '.join(required_cols)}")
                return
            
            # Process data
            with st.spinner("Processing data..."):
                player_stats, all_dates, draws_detected = load_and_process_data(df)
            
            # DEBUG: Display draws detected
            if draws_detected:
                with st.expander(f"🔍 DEBUG: {len(draws_detected)} Draw(s) Detected"):
                    st.write("The following matches were detected as draws (Score1 == Score2):")
                    for draw in draws_detected:
                        st.write(f"- **Match {draw['Match_Number']}**: {draw['Date']} | **{draw['Team1']}** vs **{draw['Team2']}** | **Score**: {draw['Score1']} - {draw['Score2']}")
            else:
                with st.expander("🔍 DEBUG: No Draws Detected"):
                    st.write("**No draws were found in your data.** This could mean:")
                    st.write("1. **All scores are different** (most likely - check your data)")
                    st.write("2. **Score parsing issues** (check if scores are stored as text or numbers)")
                    st.write("3. **Data format issues** (check if there are extra spaces or special characters)")
            
            # Display summary statistics
            st.header("📊 Summary Statistics")
            
            # Create summary dataframe
            summary_data = []
            for player in sorted(player_stats.keys()):
                stats = player_stats[player]
                summary_data.append({
                    'Player': player,
                    'Wins': stats['wins'],
                    'Draws': stats['draws'],
                    'Losses': stats['losses'],
                    'Total Points': stats['total_points'],
                    'Bonus Points': stats['bonus_points'],
                    'Total (with Bonus)': stats['total_points'] + stats['bonus_points'],
                    'Matches Played': stats['wins'] + stats['draws'] + stats['losses']
                })
            
            summary_df = pd.DataFrame(summary_data)
            # Sort by Total (with Bonus) descending
            summary_df = summary_df.sort_values(by='Total (with Bonus)', ascending=False).reset_index(drop=True)
            st.dataframe(summary_df, use_container_width=True)
            
            # Chart 1: Wins, Draws, Losses
            st.header("📈 Chart 1: Wins, Draws, and Losses per Player")
            fig1 = create_chart1_wld_stacked(player_stats)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Chart 2: Points and Bonus Points
            st.header("📈 Chart 2: Total Points and Bonus Points")
            st.markdown("**Points System:** Win = 3 pts, Draw = 1 pt, Loss = 0 pts  \n**Bonus System:** 3 wins same day = +1 bonus, 4+ wins same day = +2 bonus")
            fig2 = create_chart2_points_stacked(player_stats)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Chart 3: Timeline
            st.header("📈 Chart 3: Player Participation Timeline")
            st.markdown("🟢 **Green dot** = Played (shows W/D/L counts)  \n⭕ **Red empty dot** = Did not play")
            fig3 = create_chart3_timeline_detailed(df, player_stats, all_dates)
            st.plotly_chart(fig3, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Please upload an Excel file to get started.")
        
        # Show example data format
        with st.expander("📋 Expected Excel Format"):
            st.markdown("""
            Your Excel file should have the following columns:
            
            | Date | Court | Player1 | Player2 | Score1 | Score2 | Player3 | Player4 |
            |------|-------|---------|---------|--------|--------|---------|---------|
            | 10/02/2025 | 1 | John | Mary | 6 | 4 | Bob | Alice |
            | 10/02/2025 | 1 | John | Mary | 5 | 5 | Bob | Alice | ← This would be a DRAW
            
            - **Date**: Match date (supports DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)
            - **Court**: Court number/name
            - **Player1 + Player2**: Team 1 (Score1)
            - **Player3 + Player4**: Team 2 (Score2)
            - **Draw**: When Score1 == Score2 (e.g., 5-5, 3-3, 1-1, etc.)
            """)
            
        # Show example with draws
        with st.expander("📊 Example Data with Draws"):
            st.markdown("""
            Example data that should show draws being counted:
            
            | Date | Court | Player1 | Player2 | Score1 | Score2 | Player3 | Player4 | Result |
            |------|-------|---------|---------|--------|--------|---------|---------|--------|
            | 10/02/2025 | 1 | John | Mary | 6 | 4 | Bob | Alice | Team 1 wins |
            | 10/02/2025 | 1 | John | Mary | 5 | 5 | Bob | Alice | **DRAW** |
            | 10/02/2025 | 1 | John | Mary | 3 | 7 | Bob | Alice | Team 2 wins |
            | 10/02/2025 | 1 | Sarah | Tom | 2 | 2 | Mike | Lisa | **DRAW** |
            
            In this example, **John, Mary, Sarah, Tom, Mike, and Lisa** would all get 1 draw counted, and 1 point each.
            """)

if __name__ == "__main__":
    main()