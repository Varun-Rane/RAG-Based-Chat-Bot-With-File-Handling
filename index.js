const numbers = [12, 5, 8, 21, 3, 17, 9, 14];
const students = [
  { name: "Amit", marks: 82 },
  { name: "Riya", marks: 91 },
  { name: "Tanmay", marks: 76 }
];

function calculateAverage(arr) {
  const total = arr.reduce((sum, num) => sum + num, 0);
  return total / arr.length;
}

function findHighest(arr) {
  return Math.max(...arr);
}

function findLowest(arr) {
  return Math.min(...arr);
}

const average = calculateAverage(numbers);
const highest = findHighest(numbers);
const lowest = findLowest(numbers);

console.log("Numbers:", numbers);
console.log("Average:", average);
console.log("Highest:", highest);
console.log("Lowest:", lowest);

const passedStudents = students.filter(student => student.marks >= 80);

passedStudents.forEach(student => {
  console.log(`${student.name} scored ${student.marks}`);
});

const names = students.map(student => student.name);
console.log("Student Names:", names);

const sortedStudents = [...students].sort(
  (a, b) => b.marks - a.marks
);

console.log("Sorted Students:");

for (const student of sortedStudents) {
  console.log(student.name, student.marks);
}

let counter = 0;

while (counter < 5) {
  console.log("Counter:", counter);
  counter++;
}

const randomNumber = Math.floor(Math.random() * 100) + 1;

if (randomNumber > 50) {
  console.log("Random number is greater than 50");
} else {
  console.log("Random number is 50 or less");
}

console.log("Program completed successfully.");