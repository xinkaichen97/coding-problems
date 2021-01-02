-- Problem 06 (No. 181)
-- https://leetcode.com/problems/employees-earning-more-than-their-managers/

SELECT e1.Name As Employee 
FROM Employee e1 JOIN Employee e2
ON e2.Id = e1.ManagerId
AND e1.Salary > e2.Salary;


-- Problem 07 (No. 182)
-- https://leetcode.com/problems/duplicate-emails/
SELECT Email FROM
(SELECT Email, COUNT(Email) AS Cnt
FROM Person
GROUP BY Email)
WHERE Cnt > 1;


-- Problem 08 (No. 183)
-- https://leetcode.com/problems/customers-who-never-order/

SELECT Name AS Customers
FROM Customers LEFT JOIN Orders
ON Customers.Id = Orders.CustomerId
WHERE Orders.Id IS NULL;


-- Problem 09 (No. 184)
-- https://leetcode.com/problems/department-highest-salary/

SELECT Department, Employee, Salary
FROM 
(SELECT Department.Name AS Department, Employee.Name AS Employee, Salary, DENSE_RANK() OVER (PARTITION BY Department.Name ORDER BY Salary DESC) AS Rnk
FROM Employee JOIN Department
ON Employee.DepartmentId = Department.Id)
WHERE Rnk = 1;

-- Alternative solution (slower)
SELECT Department.Name AS Department, Employee.Name AS Employee, Salary
FROM 
(Employee JOIN Department
ON Employee.DepartmentId = Department.Id)
WHERE (Employee.DepartmentId, Salary)
IN
(SELECT DepartmentId, MAX(Salary)
FROM Employee
GROUP BY DepartmentId);


-- Problem 10 (No. 185)
-- https://leetcode.com/problems/department-top-three-salaries/

SELECT Department, Employee, Salary
FROM 
(SELECT Department.Name AS Department, Employee.Name AS Employee, Salary, DENSE_RANK() OVER (PARTITION BY Department.Name ORDER BY Salary DESC) AS Rnk
FROM Employee JOIN Department
ON Employee.DepartmentId = Department.Id)
WHERE Rnk <= 3;
