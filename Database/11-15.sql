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
