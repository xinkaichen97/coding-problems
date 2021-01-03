-- Problem 16 (No. 620)
-- https://leetcode.com/problems/not-boring-movies/

SELECT * 
FROM cinema
WHERE description <> 'boring'
AND MOD(id, 2) = 1
ORDER BY rating DESC
