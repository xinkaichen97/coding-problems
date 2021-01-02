-- Problem 11 (No. 196)
-- https://leetcode.com/problems/delete-duplicate-emails/

-- Cannot modify the same table which you use in the SELECT part, so need to use (SELECT * FROM Person) p
DELETE FROM Person
WHERE Id NOT IN
(SELECT MIN(p.Id) FROM (SELECT * FROM Person) p GROUP BY p.Email);


-- Problem 12 (No. 197)
-- https://leetcode.com/problems/rising-temperature/

SELECT w1.id AS 'Id'
FROM Weather w1, Weather w2
WHERE w1.recordDate = w2.recordDate + 1
AND w1.Temperature > w2.Temperature;


-- Problem 13 (No. 262)
-- https://leetcode.com/problems/trips-and-users/

SELECT Request_at AS "Day", ROUND(COUNT(CASE WHEN Status <> 'completed' THEN 1 END) / COUNT(*), 2) AS "Cancellation Rate"
FROM Trips
WHERE (Request_at = '2013-10-01' OR Request_at = '2013-10-02' OR Request_at = '2013-10-03')
AND Client_Id NOT IN (SELECT Users_Id FROM Users WHERE Banned = 'Yes')
AND Driver_Id NOT IN (SELECT Users_Id FROM Users WHERE Banned = 'Yes')
GROUP BY Request_at;


-- Problem 14 (No. 595)
-- https://leetcode.com/problems/big-countries/

SELECT name, population, area
FROM World
WHERE area > 3000000
OR population > 25000000


-- Problem 15 (No. 596)
-- https://leetcode.com/problems/classes-more-than-5-students/

SELECT class
FROM
(SELECT class, COUNT(DISTINCT student) student_cnt
FROM courses
GROUP BY class)
WHERE student_cnt >= 5;
