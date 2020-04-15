# NBA stats 

*The National Basketball Association (NBA) has changed since the introduction of the 3-point line back in the 1979-80 season. In the beginning, teams were reluctantly shooting 3-pointers while today, players like Steph Curry and James Harden are leading a “3-point revolution”. Basketball went from a “big guy” game to a positionless sport, based on spacing and long-range shots. But what kind of story do the stats tell? Do they agree or do they show something else?*

> Nomikos Skyllas, nomikos.skyllas@gmail.com, [Github](https://github.com/NSkyllas/NBA_stats), [LinkedIn](https://www.linkedin.com/in/nomikos-skyllas/)  


[*Click on the following image to open the interactive plot:*](https://nskyllas.github.io/NBA_Bokeh/)

[![Interactive plot](/img/Screenshot.png)](https://nskyllas.github.io/NBA_Bokeh/)
###### PCA of 17 statistical categories: Field Goal Attempts Per Game (FGA), Field Goal Percentage (FG%), 3-Point Field Goal Attempts Per Game (3PA), 3 Point Field Goal Percent-age (3P%), 2-Point Field Goal Attempts Per Game (2PA), 2-Point Field Goal Percentage (2P%), Free Throw Attempts Per Game (FTA), Free Throw Percentage (FT%), Offensive Rebounds Per Game(ORB), Defensive Rebounds Per Game (DRB), Total Rebounds Per Game (TRB), Assists Per Game(AST), Steals Per Game (STL), Blocks Per Game (BLK), Turnovers Per Game (TOV), Personal Fouls Per Game (PF) and Points Per Game (PTS). The color of each dot represents the amount of wins during the regular season (teams with more wins have brighter colors. The slider at the bottom allows to move through the seasons and see the evolution of NBA teams. The drop down menu at the top selects one NBA team and shows its position on the plot for all seasons. Vovering over the dots reveals the name of the team and the season. Panning and zooming (box and wheel) is also possible. Data collected from https://www.basketball-reference.com/.

The interactive plot is a Principal Component Analysis (PCA) of 17 stats for all NBA teams since season 1979-80 and on. A PCA plot is a very handy way to combine multiple statistical categories (points per game, 3-point shots per game etc.) and make a simple and understandable plot. The x-axis (left to right) is the most important for separating the teams (explains 38.8% of the variation in the data) and the y-axis (vertical) is slightly less important (explains 20.4% of the variation in the data). By moving the slider, teams appear from right to left unti the 2000s and from the lower left corner to the upper left, during the 2010s.  The right to left movement means that the teams evolved from an era (80s) during which 2-point shots, offensive rebounds, turnovers, personal fouls and free throws dominated to a 3-point shot dominated era. Moreover, the teams of the last decade (10s) moved from the bottom to the upper left part, meaning that in the last decade teams collected more defensive rebounds, improved their 2-point shot percentages and scored more points. The 80s teams are also higher up on the y-axis suggesting that they also had higher numbers in the aforementioned three categories compared to the 90s and 00s teams.

There are some teams worth mentioning:
- The recent Budenholzer-lead Bucks, that are clearly separating from the rest of the NBA teams. They are on the upper extreme of the plot, even further than the recent Warriors dynasty.
- The mid-00s Phoenix Suns seem like a pioneer team for its time, separating from the group of the other 00s teams and moving to the modern 3-point dominated era earlier than the rest, only to be followed almost ten years later by teams like the Rockets and the Warriors.
- The late 90s-early 00s Chicago Bulls (after Jordan’s retirement) are on the lower left extreme of the plot, separating from the other 90s-00s teams, by having very few points per game and wins per regular season.

The data suggest that the NBA has moved from a high-scoring, 2-Point and offensive rebound-dominated era with many turnovers and personal fouls to another high-scoring era (10s). This new era is dominated by 3-point shots and defensive rebounds. In between, there was a period with fewer points, fewer shots and worse shooting percentages (from mid-90s to late 00s). The data seem to follow the narrative of the ”big guy” era in the past, the ”3-point revolution” that we are witnessing today and the mid-range era in between. This is well visualized in the PCA plot where the lower right side of the plot is an ”ecosystem” in which ”big guys” can thrive by playing closer to the basket, shooting close-range 2-point shots with high percentages, collect more offensive rebounds and play tough defense (steals, personal fouls and turnovers). The lower left side is a transitional (between the 80s and the 10s) ”ecosystem” in which teams with lower scores and lower 2P% gather. This might indicate less efficient shooting decisions, possibly mid-range shots. The upper left side is ”populated” by modern teams, with more 3-point attempts, better 3-point and 2-point percentages, that collect more defensive rebounds.

<br/>
<br/>
<br/>

> - For more details read the full report: [NBA Stats Report](NBA_stats_report.md)
> - For more fun check the interactive plot: [Interactive PCA plot](https://nskyllas.github.io/NBA_Bokeh/)
> - Data collected from https://www.basketball-reference.com/
