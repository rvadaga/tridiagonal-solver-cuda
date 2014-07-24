#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<ctype.h>
#include<complex.h>
#include<math.h>
#define BUFLEN 256
#define STRLEN 10
#define PI 3.141592654

//Node datatype definition
typedef struct Node{
struct Node *prev;
struct Node *next;
char *name;
char *depname;
char *n1;
char *n2;
char *n3;
char *n4;
double value;
} Node;

//List datatype definition
typedef struct List{
Node *head;
Node *tail;
int count;
} List;

//function declaration
int main(int , char **);
void add_node(List *, Node *);
void add_node_type1(List *, char *, char *, FILE *);
void add_node_type2(List *, char *, char *, FILE *);
void add_node_type3(List *, char *, char *, FILE *);
double extract_modifier(char *, FILE *fp1);
char *check_node_name(char *, FILE *fp1);
void check_voltage_source(Node *, List *, FILE *);
void add_node_name(Node *, char **);
void search_node_name(char *, char **);
void add_node_voltage_name(char **, int);
int num_voltage_source(List *);
void add_voltage_source_current(char **, int , char **, List *, FILE *fp1);
int node_name_index(char **, char *);
int find_ground_node(char **, FILE *);
void add_conductance(List *, complex **, char **, int, char *, FILE *, double);
int check_ground_node(char *, char **, FILE *);
void add_elements_to_B(List *, complex **, char **, char **, int, FILE *);
int return_voltage_source_index(char **, char *, char *, List *, char **);
int return_voltage_source_index_1(char **, char *, List *, char **);
void add_sources(List *, int, complex *, char **, FILE *, char *, char *, double);
void full_matrix_M(complex **, complex **, complex **, int, List *);
void full_matrix_X(char **, char **, char **, int, List *);
void augmented_matrix(complex **, complex *, complex **, int, List *);
void swap_largest_element(complex **, int, int, List *, FILE *);
void row_transformation(complex **, int, int, List *, FILE *);
void solve_echelon_matrix(int, List *, complex *, complex **, FILE *);
void print_matrix_A(int, int, complex **, FILE *);
void print_matrix_B(int, int, complex **, FILE *);
void print_matrix_S(int, int, complex *, FILE *);
void print_matrix_M(int, int, complex **, FILE *);
void print_matrix_X(int, int, char **, FILE *);
void print_matrix_G(int, int, complex **, int, FILE *);
void memory_allocation_1(char **, int);
void memory_allocation_2(complex **, int, int);
void deallocate(List *, Node *, int, FILE *);
void check_voltage_source_analysis(char *, List *, FILE *);
void create_sources_for_L_and_C(List *, FILE *);
void print_solution(complex *, char **, char **, int, FILE *, int);
void print_solutions_to_file(complex *, char **, int, FILE *, char *, double, Node *);
Node *check_analysis_branch(char *, List *, FILE *);

int main(int argc, char **argv)
{//begin of main
if(argc!=2)
{
printf("Usage: ./a.out <filename>\n");
abort;
}
else
{//begin of else
FILE *fp=fopen(argv[1], "r");//creating a file pointer
if(fp == NULL)
{
printf("File could not be opened.\n");
abort;
}
else
{//begin of else - else
//variable decalaration
char *word=(char *)malloc(sizeof(char)*STRLEN);
char *pbuf=(char *)malloc(sizeof(char)*BUFLEN);
char *s=(char *)malloc(sizeof(char)*BUFLEN);
char *s1=(char *)malloc(sizeof(char)*BUFLEN);
char *analysis=(char *)malloc(sizeof(char)*STRLEN);
char *source=(char *)malloc(sizeof(char)*STRLEN);
int n, i=0, j=0, row=0, count, num_nodes, err=0, n_vs;
double vmin, vmax, vstep, wmax, wmin, num=1, value, w;
FILE *fp1=fopen("log_file.txt", "w");
FILE *fp2=fopen("spice.out", "w");
fprintf(fp1, "File is being read. A linked list will be created.\n\n");

//creating a linked list using malloc
List *list=(List *)malloc(sizeof(List));
assert(list!=NULL);
list->head=NULL;
list->count=0;
while(fgets(s, BUFLEN-1, fp))
{
s=strtok(s, " #");
if(strcmp(s, ".circuit")==0 || strcmp(s, ".circuit\n")==0)
{//checking for circuit definition beginning
fgets(s, BUFLEN-1, fp);
fprintf(fp1, "Circuit definition begins at row %d.\n\n", row+1);
break;
}
else
row++;
}
for(row=row+1; ; row++)
{                               
s=strtok(s, "#");//separating a line from its comments
strcpy(s1, s);//s1 is string without comments
count=0;
for(n=0, pbuf=s1; ; n++)
{//counting number of tokens in s1
if((word=strtok(pbuf, " \t\n"))==NULL)//extracts string tokens
break;
count++;
pbuf=NULL;
}
if(count==0)//checking for a blank line
fprintf(fp1, "Row %d is a blank line or it has only comments.\n\n", row+1);
else
{                       
printf("No. of elements in row %d - %d.\n", row+1, count);
fprintf(fp1, "No. of elements in row %d - %d.\n", row+1, count);
}
switch(count)
{
case 0://blank line
break;

case 4://node of type 1
{
//checking for node name for node type 1
if(s[0]=='r'||s[0]=='R'||s[0]=='l'||s[0]=='L'||s[0]=='c'||s[0]=='C'
||s[0]=='i'||s[0]=='I'||s[0]=='v'||s[0]=='V')
add_node_type1(list, s, pbuf, fp1);
else
{
printf("Invalid element name entered in row %d.\n\n", row+1);
fprintf(fp1, "Invalid element name entered in row %d.\n\n", row+1);
}
}
break;

case 5://node of type 2
{
//checking for node name for node type 2
if(s[0]=='f'||s[0]=='F'||s[0]=='h'||s[0]=='H')
add_node_type2(list, s, pbuf, fp1);                             
else
{
printf("Invalid element name entered in row %d.\n\n", row+1);
fprintf(fp1, "Invalid element name entered in row %d.\n\n", row+1);
}
}
break;
case 6://node of type 3
{
//checking for node name of node type 3
if(s[0]=='e'||s[0]=='E'||s[0]=='g'||s[0]=='G')
add_node_type3(list, s, pbuf, fp1);                             
else
{
printf("Invalid element name entered in row %d.\n\n", row+1);
fprintf(fp1, "Invalid element name entered in row %d.\n\n", row+1);
}
}
break;

default://none of the cases satisfy. therefore, an invalid statement
{
printf("Invalid statement entered in row %d.\n\n", row+1); 
fprintf(fp1, "Invalid statement entered in row %d.\n\n", row+1); 
}
break;
}

if(fgets(s, BUFLEN-1, fp)==NULL)//breaks when there are no more lines to read
break;

if((strcmp(s, ".end")==0) || (strcmp(s, ".end\n")==0))
//checks for end of circuit definition
{
row+=2;
fprintf(fp1, "Circuit definition ends at row %d.\n\n", row);
break;
}
}

Node *p=(Node *)malloc(sizeof(Node));
//node created for printing linked list's nodes
//finding linked list's tail
p=list->head;
while(p->next!=NULL)
{
p=p->next;
}
list->tail=p;
//printing the linked list's information in reverse order i.e.
//of the same order that has been entered 
fprintf(fp1, "The elements of the linked list are written \
in a tabular form below.\n");
fprintf(fp1, "Name      Nodes         depname   depnodes            \
value\n");
while(p!=NULL)
{
fprintf(fp1, "%-5s %6s and %-6s %6s    %6s and %-6s   %25.20E\n", 
p->name, p->n1, p->n2, p->depname, p->n3, p->n4, p->value);
p=p->prev;
}
fprintf(fp1, "\n");
strcpy(analysis, "NULL");

while(fgets(s, BUFLEN-1, fp))
{
s=strtok(s, " #");
row++;

if(strcmp(s, "\n")==0)
fprintf(fp1, "Row %d is a blank line or it has only comments.\n\n", row);

if(strcmp(s, ".analysis")==0 || strcmp(s, ".analysis\n")==0)
{//checking for circuit definition beginning
fgets(s, BUFLEN-1, fp);
fprintf(fp1, "Circuit analysis definition begins at row %d.\n\n", row);
break;
}
}

for(row=row+1; ; row++)
{                               
s=strtok(s, "#");//separating a line from its comments
strcpy(s1, s);//s1 is string without comments
count=0;

for(n=0, pbuf=s1; ; n++)
{//counting number of tokens in s1
if((word=strtok(pbuf, " \t\n"))==NULL)//extracts string tokens
break;
count++;
pbuf=NULL;
}

if(count==0)
fprintf(fp1, "Row %d is a blank line or it has only comments.\n\n", row);
else if(count==5)
{
for(n=0, pbuf=s; ; n++)
{//counting number of tokens in s1
if((word=strtok(pbuf, " \t\n"))==NULL)//extracts string tokens
break;
if(n==0)
{
strcpy(analysis, word);
if(strcmp(analysis, "dc")!=0 && strcmp(analysis, "ac")!=0)
printf("Invalid analysis type entered.\n");
}
if(n==1)
{
strcpy(source, word);
if(*source!='v' && *source!='V' && strcmp(analysis, "dc")==0)
{
printf("Invalid source name entered.\n");
err=1;
}
if(strcmp(analysis, "ac")==0)
err=0;
}
if(n==2)
{
if(strcmp(analysis, "dc")==0)
{
if(strcmp(word, "0")!=0)
{
vmin=extract_modifier(word, fp1);
if(vmin==0)
err=1;
}
else
vmin=0;          
}
if(strcmp(analysis, "ac")==0)
{
if(strcmp(word, "0")!=0)
{
wmin=2*PI*extract_modifier(word, fp1);
if(wmin<=0)
err=1;
}
else
{
wmin=0;
err=1;
}
}
}
if(n==3)
{
if(strcmp(analysis, "dc")==0)
{
if(strcmp(word, "0")!=0)
{
vmax=extract_modifier(word, fp1);
if(vmax==0)
err=1;
}
else
vmax=0;          
}
if(strcmp(analysis, "ac")==0)
{
if(strcmp(word, "0")!=0)
{
wmax=2*PI*extract_modifier(word, fp1);
if(wmax<=0)
err=1;
}
else
{
wmax=0;
err=1;
}
}
}
if(n==4)
{
if(strcmp(analysis, "dc")==0)
{
if(strcmp(word, "0")!=0)
{
vstep=extract_modifier(word, fp1);
if(vstep==0)
err=1;
}
else
{
vstep=0;
err=1;           
}
}
if(strcmp(analysis, "ac")==0)
{
if(strcmp(word, "0")!=0)
{
num=extract_modifier(word, fp1);
if(num<=0)
err=1;
}
else
{
num=0;
err=1;
}
}
}
pbuf=NULL;
}
if(strcmp(analysis, "dc")==0 && err==0)
{
check_voltage_source_analysis(source, list, fp1);
printf("Analysis: dc, source=%s, vmin=%E, vmax=%E, vstep=%E.\n\n", 
source, vmin, vmax, vstep);
fprintf(fp1, "Analysis: dc, source=%s, vmin=%E, vmax=%E, vstep=%E.\n\n", 
source, vmin, vmax, vstep);
create_sources_for_L_and_C(list, fp1);
}
if(strcmp(analysis, "ac")==0 && err==0)
{
p=check_analysis_branch(source, list, fp1);
printf("Analysis: ac, source=%s, wmin=%E, wmax=%E, n=%E.\n\n", 
source, wmin, wmax, num);
fprintf(fp1, "Analysis: ac, source=%s, wmin=%E, wmax=%E, n=%E.\n\n", 
source, wmin, wmax, num);
}
if(err==1)
{       
printf("Invalid circuit analysis statement has been given.\n\n");
fprintf(fp1, "Invalid circuit analysis statement has been given.\n\n");
printf("Program will terminate.\n\n");
fprintf(fp1, "Program will terminate.\n\n");
exit(8);
}
}

if(fgets(s, BUFLEN-1, fp)==NULL)//breaks when there are no more lines to read
break;
if((strcmp(s, ".end")==0) || (strcmp(s, ".end\n")==0))
//checks for end of circuit analysis definition
{
if(strcmp(analysis, "NULL")==0)
{
printf("No circuit anaylsis statement has been given.\n\n");
printf("Program will terminate.\n\n");
fprintf(fp1, "No circuit anaylsis statement has been given.\n\n");
fprintf(fp1, "Program will terminate.\n\n");
exit(8);
row+=1;
}
fprintf(fp1, "Circuit analysis definition ends at row %d.\n\n", row);
break;
}
}

char **node_name=(char **)malloc(sizeof(char *)*(2*list->count));
//node_name to store distinct node names
*(node_name)=(char *)malloc(sizeof(char)*STRLEN);
strcpy(*(node_name), "NULL"); 
value=1/num;
value=pow(10, value);

//checking for distinct nodes and adding them to node_name using add_node_name
p=list->head;     
while(p!=NULL)
{
add_node_name(p, node_name);
if(*p->name=='h'|| *p->name=='H'|| *p->name=='f'|| *p->name=='F')
check_voltage_source(p, list, fp1);
p=p->next;
}
i=0;
while(strcmp(*(node_name+i), "NULL")!=0)
{
i++;
}
num_nodes=i;
n_vs=num_voltage_source(list);

//swapping ground node to last position element in node_name array
i=0;
if(strcmp(*(node_name+num_nodes-1), "0")!=0)
{
while(strcmp(*(node_name+i), "0")!=0)
{
i++;
}
strcpy(*(node_name+i), *(node_name+num_nodes-1));
strcpy(*(node_name+num_nodes-1), "0");
}

//printing the distinct nodes list      
i=0;
fprintf(fp1, "Array of distinct node names:\n");
while(strcmp(*(node_name+i), "NULL")!=0)
{
fprintf(fp1, "[%d] :: %-s\n", i, *(node_name+i));
i++;
}
fprintf(fp1, "Ground node found at index %d.\n", find_ground_node(node_name, fp1));
fprintf(fp1, "\n");

//creating node voltage vector - vector V bar
char **v=(char **)malloc(sizeof(char *)*num_nodes);
memory_allocation_1(v, num_nodes);

//creating vector I bar for currents through voltage source
char **c=(char **)malloc(sizeof(char *)*n_vs);
memory_allocation_1(c, num_nodes);              

//creating matrix A for conductance.
complex **A=(complex **)malloc((sizeof(complex *)*(num_nodes+n_vs)));
memory_allocation_2(A, num_nodes+n_vs, num_nodes+n_vs);         

//creating matrix B for voltage source currents
complex **B=(complex **)malloc((sizeof(complex *)*(num_nodes+n_vs)));
memory_allocation_2(B, num_nodes+n_vs, n_vs);

//creating source vector S for current sources and voltage sources.
complex *S=(complex *)malloc(sizeof(complex)*(num_nodes+n_vs));

//creating full matrix M for storing matrices A and B
complex **M=(complex **)malloc((sizeof(complex *)*(num_nodes+n_vs)));
memory_allocation_2(M, num_nodes+n_vs, num_nodes+n_vs);

//creating unknown variables matrix x
char **x=(char **)malloc(sizeof(char *)*(num_nodes+n_vs));
memory_allocation_1(x, num_nodes+n_vs); 

//creating augmented matrix G 
complex **G=(complex **)malloc((sizeof(complex *)*(num_nodes+n_vs)));
memory_allocation_2(G, num_nodes+n_vs, num_nodes+n_vs+1);

//creating solution matrix H
complex *H=(complex *)malloc(sizeof(complex)*(num_nodes+n_vs));

fprintf(fp1, "The equation takes the following form.\n");
fprintf(fp1, "matrix(A)*matrix(V)+matrix(B)*matrix(I)=matrix(S).\n\n");

//adding node voltages to the vector V bar
add_node_voltage_name(v, num_nodes);
fprintf(fp1, "Matrix V has dimensions %d*%d.\n", num_nodes, 1);
fprintf(fp1, "The Vector V bar is:\n");
for(i=0; i<num_nodes; i++)
{
fprintf(fp1, "%s\n", v[i]);
}
fprintf(fp1, "\n");

//setting the fields of vector I bar as appropriate
add_voltage_source_current(c, n_vs, node_name, list, fp1);
fprintf(fp1, "Matrix I has dimensions %d*1.\n", n_vs);
fprintf(fp1, "The Vector I bar is:\n");
for(i=0; i<n_vs; i++)
{
fprintf(fp1, "%s\n", c[i]);
}
fprintf(fp1, "\n");

if(strcmp(analysis, "dc")==0)
{ 
value=vmin;

//storing elements in matrix A
add_conductance(list, A, node_name, num_nodes, analysis, fp1, 0);
print_matrix_A(num_nodes, n_vs, A, fp1); 

//storing elements in matrix B
add_elements_to_B(list, B, c, node_name, num_nodes, fp1);
print_matrix_B(num_nodes, n_vs, B, fp1);

//adding elements to X in full matrix form
full_matrix_X(v, c, x, num_nodes, list);
print_matrix_X(num_nodes, n_vs, x, fp1);

//adding elements to M in full matrix form
full_matrix_M(A, B, M, num_nodes, list);
print_matrix_M(num_nodes, n_vs, M, fp1);

fprintf(fp1, "The equation has been rewritten as: matrix(M)*matrix(x)=matrix(S)\n\n");
fprintf(fp2, "#%14s", source);

print_solution(H, x, node_name, num_nodes+n_vs, fp2, 3);

while(value<=vmax)
{
fprintf(fp1, "This is being done for value of %s=%E.\n\n", source, value);

//adding elements to source vector
complex *S=(complex *)malloc(sizeof(complex)*(num_nodes+n_vs));
add_sources(list, num_nodes, S, node_name, fp1, analysis, source, value);
print_matrix_S(num_nodes, n_vs, S, fp1);

//adding elements to G in augmented matrix form
augmented_matrix(M, S, G, num_nodes, list);
fprintf(fp1, "Matrix G has dimensions %d*%d.\n", num_nodes+n_vs, num_nodes+n_vs+1);
print_matrix_G(num_nodes, n_vs, G, 0, fp1);

//gaussian elimination by performing pivotting and row transformations
for(i=0; i<num_nodes+n_vs; i++)
{
swap_largest_element(G, num_nodes, i, list, fp1);
row_transformation(G, num_nodes, i, list, fp1); 
}
fprintf(fp1, "\n");

print_matrix_G(num_nodes, n_vs, G, 1, fp1);

//solving the echelon form matrix
solve_echelon_matrix(num_nodes, list, H, G, fp1);

//printing the solutions
print_solution(H, x, node_name, num_nodes+n_vs, fp1, 1);
print_solutions_to_file(H, node_name, (num_nodes+n_vs), fp2, analysis, value, NULL);
value+=vstep;
}
printf("Data written in tabular form in the file \"spice.out\".\n\n");
fprintf(fp1, "Data written in tabular form in the file \"spice.out\".\n\n");
}

if(strcmp(analysis, "ac")==0)
{          
w=wmin;
p=check_analysis_branch(source, list, fp1);

//storing elements in matrix B
add_elements_to_B(list, B, c, node_name, num_nodes, fp1);
print_matrix_B(num_nodes, n_vs, B, fp1);

//adding elements to X in full matrix form
full_matrix_X(v, c, x, num_nodes, list);
print_matrix_X(num_nodes, n_vs, x, fp1);

//adding elements to source vector
add_sources(list, num_nodes, S, node_name, fp1, analysis, source, p->value);
print_matrix_S(num_nodes, n_vs, S, fp1);

fprintf(fp2, "#%14s    |V[%s]-V[%s]|    <|(V[%s]-V[%s])\n", 
"w", p->n1, p->n2, p->n1, p->n2);

while(w<=wmax)
{
fprintf(fp1, "This is for w=%E.\n", w);
//storing elements in matrix A
memory_allocation_2(A, num_nodes+n_vs, num_nodes+n_vs);
add_conductance(list, A, node_name, num_nodes, analysis, fp1, w);
print_matrix_A(num_nodes, n_vs, A, fp1); 

//adding elements to M in full matrix form
full_matrix_M(A, B, M, num_nodes, list);
print_matrix_M(num_nodes, n_vs, M, fp1);

//adding elements to G in augmented matrix form
augmented_matrix(M, S, G, num_nodes, list);

fprintf(fp1, "Matrix G has dimensions %d*%d.\n", num_nodes+n_vs, num_nodes+n_vs+1);
print_matrix_G(num_nodes, n_vs, G, 0, fp1);

//gaussian elimination by performing pivotting and row transformations
for(i=0; i<num_nodes+n_vs; i++)
{
swap_largest_element(G, num_nodes, i, list, fp1);
row_transformation(G, num_nodes, i, list, fp1);                            
}
fprintf(fp1, "\n");

print_matrix_G(num_nodes, n_vs, G, 1, fp1);

//solving the echelon form matrix
solve_echelon_matrix(num_nodes, list, H, G, fp1);

//printing the solutions
print_solution(H, x, node_name, num_nodes+n_vs, fp1, 1);
print_solutions_to_file(H, node_name, (num_nodes+n_vs), fp2, analysis, w, p);
w=w*value;
}
printf("Data written in tabular form in the file \"spice.out\".\n");
fprintf(fp1, "Data written in tabular form in the file \"spice.out\".\n");
}

printf("Log file \"log_file.txt\" has been created to \
maintain a record of the various\noperations that have \
been done to find the solution.\n"); 
printf("In order to see the log file \"log_file.txt\" \
clearly, deselect the text wrapping mode.\n\n");

//deallocation of memory allocated to the linked list

fprintf(fp1, "Deallocation of memory will be carried out now.\n");
deallocate(list, list->head, list->count, fp1);
free(p);
free(G);
free(list->head);
fprintf(fp1, "The memory of node at position 1 has been freed.\n");
free(list);
fprintf(fp1, "The linked list's memory has been deallocated.\n\n");
}//end of else - else
//closing the file pointer
fclose(fp);             
}//end of else
return 0;
}//end of main

void add_node(List *list, Node *node)
{
if(list->head!=NULL)//for assigning the remaining nodes
{
list->head->prev=node;
//assigning prev ptr of list head to node being added 
node->prev=NULL;
//assigning prev of node being added to NULL because it'll be the new list head         
node->next=list->head;
//new node's next is pointing to old list head node 
list->head=node;
//new node made the lists's head
(list->count)+=1;
//incrementing list's count
}
else//for assigning head of the list
{
list->head=node;
//assigns head to node being added
list->head->prev=NULL;
//assigns prev ptr of list head to NULL
(list->count)+=1;
//increments list's count
}
}

//node of type 1 being added to linked list after checking it
void add_node_type1(List *list, char *s, char *pbuf, FILE *fp1)
{
//variable declaration
int n, c=0;
char *word=(char *)malloc(sizeof(char)*STRLEN);
Node *node=(Node *)malloc(sizeof(Node)); 
assert(node!=NULL);
for(n=0, pbuf=s; ; n++)
{
//tokenizing the string s
if((word=strtok(pbuf, " \t\n"))==NULL)
break;
if(n==0)
{//element's name
node->name=(char *)malloc(sizeof(char)*STRLEN);
strcpy(node->name, word);
}
if(n==1)
{//element's from node
node->n1=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n1, check_node_name(word, fp1));
} 
if(n==2)
{//element's to node
node->n2=(char *)malloc(sizeof(char)*STRLEN);
strcpy(node->n2, check_node_name(word, fp1));
}
if(n==3)
{//element's value. It's assumed not to be 0 except for Voltage source.
if(strcmp(word, "0")!=0) 
node->value=extract_modifier(word, fp1);
//modifying the value based on following character
else
node->value=0; 
}
pbuf=NULL;
}
//node printed after checking validity of it's members
if((node->value!=0 && strcmp(node->n1, "invalid")!=0 && strcmp(node->n2, "invalid")!=0) 
|| ((*node->name=='V' || *node->name=='v') && strcmp(node->n1, "invalid")!=0 && 
strcmp(node->n2, "invalid")!=0) )
{
node->n4=node->depname=node->n3=NULL;
add_node(list, node);
printf("Node name: %s, nodes: %s and %s, value: %E\n\n", node->name, 
node->n1, node->n2, node->value);
fprintf(fp1, "Node name: %s, nodes: %s and %s, value: %E\n\n", node->name, 
node->n1, node->n2, node->value);
}
//node is freed if any one of the element's member is invalid
else
{
printf("\n");
fprintf(fp1, "\n");
free(node);
}
}

//node of type 2 being added to linked list after checking it
void add_node_type2(List *list, char *s, char *pbuf, FILE *fp1)
{
//variable declaration
int n;
char *word=(char *)malloc(sizeof(char)*STRLEN);
Node *node=(Node *)malloc(sizeof(Node));                                
assert(node!=NULL);
for(n=0, pbuf=s; ; n++)
{
//tokenizing the string s
if((word=strtok(pbuf, " \t\n"))==NULL)
break;
if(n==0)
{//element's name                                       
node->name=(char *)malloc(sizeof(char)*STRLEN);
strcpy(node->name, word);
}
if(n==1)
{//element's from node
node->n1=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n1, check_node_name(word, fp1));
}                                       
if(n==2)
{//element's to node
node->n2=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n2, check_node_name(word, fp1));
}
if(n==3)
{//element's depname
node->depname=(char *)malloc(sizeof(char)*STRLEN);
//element should depend on current through a voltage source 
if(*(word+0)=='V' || *(word+0)=='v')
strcpy(node->depname, word);
else
{
printf("Wrong dependent name entered.\n");
fprintf(fp1, "Wrong dependent name entered.\n");
node->depname=NULL;
}
}
if(n==4)
{//element's value. It's assumed not to be 0.
if(strcmp(word, "0")!=0)
node->value=extract_modifier(word, fp1);
//modifying the value based on following character
else
node->value=0;                  
}
pbuf=NULL;
}
//node printed after checking validity of it's members
if(node->value!=0 && node->depname!=NULL && strcmp(node->n1, "invalid")!=0 && 
strcmp(node->n2, "invalid")!=0)
{
node->n4=node->n3=NULL;
add_node(list, node);
printf("Node name: %s, nodes: %s and %s, depname: %s, value: %E\n\n", 
node->name, node->n1, node->n2, node->depname, node->value);
fprintf(fp1, "Node name: %s, nodes: %s and %s, depname: %s, value: %E\n\n", 
node->name, node->n1, node->n2, node->depname, node->value);
}
//node is freed if any one of the element's member is invalid
else
{
printf("\n");
fprintf(fp1, "\n");
free(node);
}
}

//node of type 3 being added to linked list after checking it
void add_node_type3(List *list, char *s, char *pbuf, FILE *fp1)
{
//variable declaration
int n;
char *word=(char *)malloc(sizeof(char)*STRLEN);
Node *node=(Node *)malloc(sizeof(Node));                                
assert(node!=NULL);
for(n=0, pbuf=s; ; n++)
{
//tokenizing the string s
if((word=strtok(pbuf, " \t\n"))==NULL)
break;
if(n==0)
{//element's name                                               
node->name=(char *)malloc(sizeof(char)*STRLEN);
strcpy(node->name, word);
}
if(n==1)
{//element's from node
node->n1=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n1, check_node_name(word, fp1));
}                                       
if(n==2)
{//element's to node
node->n2=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n2, check_node_name(word, fp1));
}
if(n==3)
{//from node on whose voltage, element is dependent
node->n3=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n3, check_node_name(word, fp1));
}
if(n==4)
{//to node on whose voltage, element is dependent
node->n4=(char *)malloc(sizeof(char)*STRLEN);
//if node name is valid then it's copied to link's from node
strcpy(node->n4, check_node_name(word, fp1));
}
if(n==5)
{//element's value. It's assumed not to be 0.
if(strcmp(word, "0")!=0)
node->value=extract_modifier(word, fp1);
//modifying the value based on following character
else
node->value=0;                  
}
pbuf=NULL;
}
//node printed after checking validity of it's members
if(node->value!=0 && strcmp(node->n1, "invalid")!=0 && strcmp(node->n2, "invalid")!=0 
&& strcmp(node->n3, "invalid")!=0 && strcmp(node->n4, "invalid")!=0)
{
node->depname=NULL;
add_node(list, node);
printf("Node name: %s, nodes: %s and %s, depnodes: %s and %s, value: %E\n\n", 
node->name, node->n1, node->n2, node->n3, node->n4, node->value);
fprintf(fp1, "Node name: %s, nodes: %s and %s, depnodes: %s and %s, value: %E\n\n", 
node->name, node->n1, node->n2, node->n3, node->n4, node->value);
}
//node is freed if any one of the element's member is invalid
else
{
printf("\n");
fprintf(fp1, "\n");
free(node);
}
}

//function to modify the value based on it's following character(s)
double extract_modifier(char *word, FILE *fp1)
{
char *c=(char *)malloc(sizeof(char)*STRLEN);
int a;
double var;
//a is used to receive number of arguments or tokens
a=sscanf(word, "%lf%c%c%c", &var, c, c+1, c+2);
//modifying value if character is n
if(a==2 && *c=='n')
return (var*(1.00E-009));
//modifying value if character is u
else if(a==2 && *c=='u')
return (var*(1.00E-006));
//modifying value if character is m
else if(a==2 && *c=='m')
return (var*(1.00E-003));
//modifying value if character is k
else if(a==2 && *c=='k')
return (var*(1.00E+003));
//modifying value if characters are m e g 
else if(a==4 && *c=='m' && *(c+1)=='e' && *(c+2)=='g')
return (var*(1.00E+006));
//if there are no characters, then same value is returned
else if(a==1)
return var;
//error message
else
{
printf("Wrong value entered.\n");
fprintf(fp1, "Wrong value entered.\n");
return 0;
}
}

//function to check validity of node name
char *check_node_name(char *word, FILE *fp1)
{
int i=0, n=0;
char *c=(char *)malloc(sizeof(char)*STRLEN);
strcpy(c, word);
while(c[i]!='\0')
{
//checking if each character is an alphabet or a digit or a hyphen or an underscore
if(isalnum(c[i]) || *(c+i)=='_' || *(c+i)=='-')
{
n++;
i++;
}
else 
break;
}
//checking if node name begin;s with N or n and if there
//are no special characters or not
if(n==strlen(c) && (c[0]=='n' || c[0]=='N'))
return word;
//node name valid if it is a digit
else if(n==strlen(c) && isdigit(c[0]))
return word;
//error message
else
{
c="invalid";
printf("Wrong node name entered.\n");
fprintf(fp1, "Wrong node name entered.\n");
return c;
}
}

//function to check if depname in current controlled source is valid or not
void check_voltage_source(Node *p, List *list, FILE *fp1)
{
int c=0;
Node *node=(Node *)malloc(sizeof(Node *));
node=list->head;
while(node!=NULL)
{
if(strcmp(node->name, p->depname)==0)
c=1;
node=node->next;
}
if(c==0)
{
printf("Voltage source %s given in current controlled source %s \
does not exist.\n", p->depname, p->name);
printf("Program will terminate.\n\n");
fprintf(fp1,"Voltage source %s given in current controlled source %s \
does not exist.\n", p->depname, p->name);
fprintf(fp1, "Program will terminate.\n\n");
exit(8);
}
}

//function to provide each node name to add it to distinct nodes 
//array if it does not repeat
void add_node_name(Node *p, char **node_name)
{
int c=0, i=0;
//checking type of node
if(p->depname==NULL && p->n3==NULL && p->n4==NULL)
c=4;
else if(p->depname==NULL)
c=6;
else
c=5;
//for type 1 and 2 there are only 3 members with node names specified
//search_node_name function is used to check if that node name is already 
//present and adds it if it's not present
if(c==4 || c==5)
{
search_node_name(p->n1, node_name);
search_node_name(p->n2, node_name);
}
//for type 3 nodes there are 4 members with node names specified
if(c==6)
{
search_node_name(p->n1, node_name);
search_node_name(p->n2, node_name);
search_node_name(p->n3, node_name);
search_node_name(p->n4, node_name);
}
}

//function to check if a given node name is existing or not.
//If it doesn't, it adds that node name to distinct node's array 
void search_node_name(char *str, char **node_name)
{
int i=0, c=1;
if(strcmp(*(node_name), "NULL")==0)
{
strcpy(*(node_name), str);
*(node_name+1)=(char *)malloc(sizeof(char)*STRLEN);
strcpy(*(node_name+1), "NULL");
}
while(strcmp(*(node_name+i), "NULL")!=0)
{
if(strcmp(*(node_name+i), str)==0)
c=0;
i++;
}
if(c==1 && i!=0)
{
strcpy(*(node_name+i), str);
*(node_name+i+1)=(char *)malloc(sizeof(char)*STRLEN);
strcpy(*(node_name+i+1), "NULL");
}
}


//function to add node voltage's name to vector A
void add_node_voltage_name(char **v, int n)
{
int i;
//each element in array v is changed based on the index of the node name.
//e.g. if 5 is number assigned to node "n3", then it sets the element in index 5 to V[5]
for(i=0; i<n; i++)
{
sprintf(*(v+i), "V[%d]", i);
}
}

int num_voltage_source(List *list)
{
int c=0;
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{ 
if(*p->name=='V' || *p->name=='v' ||*p->name=='E' || *p->name=='e' || 
*p->name=='H' || *p->name=='h')
c++;
p=p->next;
}
return c;
free(p);
}

//function to add current unknown in vector B 
//based on the voltage source on which the element is dependent on. 
void add_voltage_source_current(char **c, int n, char **node_name, List *list, FILE *fp1)
{
int i=0;
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{
if(*(p->name)=='V' || *(p->name)=='v')
{
sprintf(*(c+i), "I[%d][%d]", node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2));
fprintf(fp1, "For voltage source %s, current I[%d][%d] has been added \
at row %d.\n", p->name, node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), i);
i++;
}

if(*(p->name)=='E' || *(p->name)=='e')
{
sprintf(*(c+i), "I[%d][%d]", node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2));
fprintf(fp1, "For VCVS %s, current I[%d][%d] has been added at \
row %d.\n", p->name, node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), i);
i++;
}

if(*(p->name)=='H' || *(p->name)=='h')
{
sprintf(*(c+i), "I[%d][%d]", node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2));
fprintf(fp1, "For CCVS %s, current I[%d][%d] has been added at \
row %d.\n", p->name, node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), i);
i++;
}
p=p->next;
}
fprintf(fp1, "\n");
free(p);
}

//function to find the index of a given node name
int node_name_index(char **node_name, char *name)
{
int i=0;
while(strcmp(*(node_name+i), "NULL")!=0)
{
if(strcmp(*(node_name+i), name)==0)
return i;
i++;
}
}

//function to find a ground node which has name "0", and to 
//return the value of the index in the node_name array corresponding
//to the ground node
int find_ground_node(char **node_name, FILE *fp1)
{
int i=0;
while(strcmp(*(node_name+i), "NULL")!=0)
{
if(strcmp(*(node_name+i), "0")==0)
{
return i;
}
i++;
} 
//If no ground node has been specified, it print's error message and 
//terminates the program.
if(strcmp(*(node_name), "0")!=0)
{
printf("\nGround node has not been specified.\nProgram will terminate.\n\n");
fprintf(fp1, "\nGround node has not been specified.\nProgram will terminate.\n\n");
exit(8);
}
}

//function to add elements of conductance matrix : matrix A
void add_conductance(List *list, complex **A, char **node_name, 
int num_nodes, char *analysis, FILE *fp1, double w)
{
int i=1, n_vs=num_voltage_source(list);
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{
if(*p->name=='R' || *p->name=='r')
{
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n1)]+=1/p->value;
fprintf(fp1, "A[%d][%d]: For resistor %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n1), 
p->name, p->n1, 1/p->value, node_name_index(node_name, p->n1), p->n1);
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n2)]+=-1/p->value;
fprintf(fp1, "A[%d][%d]: For resistor %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n2), 
p->name, p->n1, -1/p->value, node_name_index(node_name, p->n2), p->n2);   
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n2)]+=1/p->value;
fprintf(fp1, "A[%d][%d]: For resistor %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n2), 
p->name, p->n2, 1/p->value, node_name_index(node_name, p->n2), p->n2);   
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n1)]+=-1/p->value;
fprintf(fp1, "A[%d][%d]: For resistor %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n1), 
p->name, p->n2, -1/p->value, node_name_index(node_name, p->n1), p->n1);   
}

if((*p->name=='L' || *p->name=='l') && strcmp(analysis, "ac")==0)
{
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n1)]+=-I/(w*p->value);
fprintf(fp1, "A[%d][%d]: For inductor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n1), 
p->name, p->n1, -1/(w*p->value), node_name_index(node_name, p->n1), p->n1);
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n2)]+=I/(w*p->value);
fprintf(fp1, "A[%d][%d]: For inductor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n2), 
p->name, p->n1, 1/(w*p->value), node_name_index(node_name, p->n2), p->n2);   
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n2)]+=-I/(w*p->value);
fprintf(fp1, "A[%d][%d]: For inductor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n2), 
p->name, p->n2, -1/(w*p->value), node_name_index(node_name, p->n2), p->n2);
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n1)]+=I/(w*p->value);
fprintf(fp1, "A[%d][%d]: For inductor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n1), 
p->name, p->n2, 1/(w*p->value), node_name_index(node_name, p->n1), p->n1);
}

if((*p->name=='C' || *p->name=='c') && strcmp(analysis, "ac")==0)
{
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n1)]+=(p->value)*w*I;
fprintf(fp1, "A[%d][%d]: For capacitor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n1), 
p->name, p->n1, (p->value)*w, node_name_index(node_name, p->n1), p->n1);
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n2)]+=(-p->value)*w*I;
fprintf(fp1, "A[%d][%d]: For capacitor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n2), 
p->name, p->n1, -(p->value)*w, node_name_index(node_name, p->n2), p->n2);
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n2)]+=(p->value)*w*I;
fprintf(fp1, "A[%d][%d]: For capacitor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n2), 
p->name, p->n2, (p->value)*w, node_name_index(node_name, p->n2), p->n2);
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n1)]+=(-p->value)*w*I;
fprintf(fp1, "A[%d][%d]: For capacitor %s, at the node %s, the value %13.5Ej \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n1), 
p->name, p->n2, -(p->value)*w, node_name_index(node_name, p->n1), p->n1);
}

if(*p->name=='G' || *p->name=='g')
{
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n3)]+=-p->value;
fprintf(fp1, "A[%d][%d]: For VCCS %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n3), 
p->name, p->n1, -p->value, node_name_index(node_name, p->n3), p->n3);
A[node_name_index(node_name, p->n1)][node_name_index(node_name, p->n4)]+=p->value;
fprintf(fp1, "A[%d][%d]: For VCCS %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n4), 
p->name, p->n1, p->value, node_name_index(node_name, p->n4), p->n4);
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n3)]+=p->value;
fprintf(fp1, "A[%d][%d]: For VCCS %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n3), 
p->name, p->n2, p->value, node_name_index(node_name, p->n3), p->n3);
A[node_name_index(node_name, p->n2)][node_name_index(node_name, p->n4)]+=-p->value;
fprintf(fp1, "A[%d][%d]: For VCCS %s, at the node %s, the value %13.5E \
has been added corresponding to voltage V[%d] at node %s.\n", 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n4), 
p->name, p->n2, -p->value, node_name_index(node_name, p->n4), p->n4);
}

if(*p->name=='V' || *p->name=='v')
{
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n1)]+=1;
fprintf(fp1, "A[%d][%d]: For voltage source %s,  1 has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n1), p->name, node_name_index(node_name, p->n1), p->n1); 
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n2)]+=-1;
fprintf(fp1, "A[%d][%d]: For voltage source %s, -1 has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n2), p->name, node_name_index(node_name, p->n2), p->n2);
i++;
}

if(*p->name=='E' || *p->name=='e')
{
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n1)]+=1;
fprintf(fp1, "A[%d][%d]: For VCVS %s,  1 has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n1), p->name, node_name_index(node_name, p->n1), p->n1);
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n2)]+=-1;
fprintf(fp1, "A[%d][%d]: For VCVS %s, -1 has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n2), p->name, node_name_index(node_name, p->n2), p->n2);
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n3)]+=-p->value;
fprintf(fp1, "A[%d][%d]: For VCVS %s, %E has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n3), p->name, -p->value, 
node_name_index(node_name, p->n3), p->n3);
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n4)]+=p->value;
fprintf(fp1, "A[%d][%d]: For VCVS %s, %E has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n4), p->name, p->value, 
node_name_index(node_name, p->n4), p->n4);
i++;
}

if(*p->name=='H' || *p->name=='h')
{
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n1)]+=1;
fprintf(fp1, "A[%d][%d]: For CCVS %s, 1 has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n1), p->name, 
node_name_index(node_name, p->n1), p->n1); 
A[node_name_index(node_name, "0")+i][node_name_index(node_name, p->n2)]+=-1;
fprintf(fp1, "A[%d][%d]: For CCVS %s, -1 has been added for voltage V[%d] \
corresponding at node %s.\n", node_name_index(node_name, "0")+i, 
node_name_index(node_name, p->n2), p->name, 
node_name_index(node_name, p->n2), p->n2);
i++;
}
p=p->next;
}
for(i=0; i<node_name_index(node_name, "0"); i++)
{
A[node_name_index(node_name, "0")][i]=0;
}
A[node_name_index(node_name, "0")][node_name_index(node_name, "0")]=1;
fprintf(fp1, "A[%d]: The ground node equation for node %s \
has been written at row %d.\n", node_name_index(node_name, "0"), "0", 
node_name_index(node_name, "0"));
fprintf(fp1, "\n");
free(p);
}

//function to check whether a given node is a ground node or not
//If it is a ground node, it returns 1 else 0.
int check_ground_node(char *s, char **node_name, FILE *fp1)
{
int i=0, c=0;
while(strcmp(*(node_name+i), "NULL")!=0)
{
if(strcmp(*(node_name+find_ground_node(node_name, fp1)), s)==0)
{
c=1;
return 1;
}
i++;
}
if(c==0)
return 0;
}

//function to add elements to B
void add_elements_to_B(List *list, complex **B, char **c, char **node_name, 
int num_nodes, FILE *fp1)
{
int i, n_vs=num_voltage_source(list);
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{
if(*p->name=='v' || *p->name=='V')
{
B[node_name_index(node_name, p->n1)]
[return_voltage_source_index(c, p->n1, p->n2, list, node_name)]+=1;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value  1 has been added \
corresponding to current I[%d][%d] for voltage source %2s.\n", 
node_name_index(node_name, p->n1), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n1,
node_name_index(node_name, p->n1), node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->name);
B[node_name_index(node_name, p->n2)]
[return_voltage_source_index(c, p->n1, p->n2, list, node_name)]+=-1;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value -1 has been added \
corresponding to current I[%d][%d] for voltage source %2s.\n", 
node_name_index(node_name, p->n2), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n2, 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->name);
}

if(*p->name=='e' || *p->name=='E')
{
B[node_name_index(node_name, p->n1)]
[return_voltage_source_index(c, p->n1, p->n2, list, node_name)]+=1;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value  1 has been added \
corresponding to current I[%d][%d] for VCVS %2s.\n", 
node_name_index(node_name, p->n1), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n1, 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->name);
B[node_name_index(node_name, p->n2)]
[return_voltage_source_index(c, p->n1, p->n2, list, node_name)]+=-1;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value -1 has been added \
corresponding to current I[%d][%d] for VCVS %2s.\n", 
node_name_index(node_name, p->n2), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n2, 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->name);
}

if(*p->name=='f' || *p->name=='F')
{
B[node_name_index(node_name, p->n1)]
[return_voltage_source_index_1(c, p->depname, list, node_name)]+=-p->value;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value %E has been added \
corresponding to current I[%d][%d] for CCCS %2s.\n", 
node_name_index(node_name, p->n1), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n1, 
node_name_index(node_name, p->n1), -p->value, node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->depname);
B[node_name_index(node_name, p->n2)]
[return_voltage_source_index_1(c, p->depname, list, node_name)]+=p->value;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value %E has been added \
corresponding to current I[%d][%d] for CCCS %2s.\n", 
node_name_index(node_name, p->n2), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n2, 
node_name_index(node_name, p->n2), p->value, node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->depname);
}

if(*p->name=='h' || *p->name=='H')
{
B[node_name_index(node_name, p->n1)]
[return_voltage_source_index_1(c, p->depname, list, node_name)]+=1;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value 1 has been added \
corresponding to current I[%d][%d] for CCVS %2s.\n", 
node_name_index(node_name, p->n1), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n1, 
node_name_index(node_name, p->n1), node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->depname);
B[node_name_index(node_name, p->n2)]
[return_voltage_source_index_1(c, p->depname, list, node_name)]+=-1;
fprintf(fp1, "B[%d][%d]: At node %2s with index %d, a value -1 has been added \
corresponding to current I[%d][%d] for CCVS %2s.\n", 
node_name_index(node_name, p->n2), 
return_voltage_source_index(c, p->n1, p->n2, list, node_name), p->n2, 
node_name_index(node_name, p->n2), node_name_index(node_name, p->n1), 
node_name_index(node_name, p->n2), p->depname);
}
p=p->next;
}
for(i=0; i<n_vs; i++)
{
B[node_name_index(node_name, "0")][i]=0;
}
fprintf(fp1, "B[%d]: The ground node equation has been written at \
row %d.\n", node_name_index(node_name, "0"), node_name_index(node_name, "0"));
fprintf(fp1, "\n");
free(p);
}

//function to return index of the current of a given voltage source
int return_voltage_source_index(char **c, char *s1, char *s2, List *list, 
char **node_name)
{
int i, a, b, n_vs=num_voltage_source(list);
for(i=0; i<n_vs; i++)
{
sscanf(*(c+i), "I[%d][%d]", &a, &b);
if(a==node_name_index(node_name, s1) && b==node_name_index(node_name, s2))
return i;
}
}

//function to return index of the current of a given voltage source 
//given the voltage source name
int return_voltage_source_index_1(char **c, char *depname, List *list, 
char **node_name)
{
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{
if((*p->name=='V' || *p->name=='v') && strcmp(depname, p->name)==0)
return return_voltage_source_index(c, p->n1, p->n2, list, node_name);
p=p->next;
}
}

//function to add sources to S vector
void add_sources(List *list, int num_nodes, complex *S, char **node_name, 
FILE *fp1, char *analysis, char *source, double value)
{
int i, n_vs=num_voltage_source(list);
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
for(i=0; i<num_nodes+n_vs; i++)
{
S[i]=0;
}
i=0;
while(p!=NULL)
{
if(*p->name=='I' || *p->name=='i')
{
if(check_ground_node(p->n1, node_name, fp1)==0)
{
S[node_name_index(node_name, p->n1)]+=p->value;
fprintf(fp1, "S[%d]: At node %2s with index %d, the value %13.5E has been added \
corresponding to current source %2s.\n", node_name_index(node_name, p->n1), p->n1, 
node_name_index(node_name, p->n1), p->value, p->name);
}
if(check_ground_node(p->n2, node_name, fp1)==0)
{
S[node_name_index(node_name, p->n2)]+=-p->value;      
fprintf(fp1, "S[%d]: At node %2s with index %d, the value %13.5E has been added \
corresponding to current source %2s.\n", node_name_index(node_name, p->n2), p->n2, 
node_name_index(node_name, p->n2), -p->value, p->name);
}
}

if(*p->name=='V' || *p->name=='v')
{
if(strcmp(analysis, "dc")==0 && strcmp(source, p->name)==0)
{
S[node_name_index(node_name, "0")+1+i]+=value;
fprintf(fp1, "S[%d]: For voltage source %s, the value at row %d has been added \
by %E.\n", node_name_index(node_name, "0")+1+i, p->name, 
node_name_index(node_name, "0")+1+i, value);
}
else
{
S[node_name_index(node_name, "0")+1+i]+=p->value;
fprintf(fp1, "S[%d]: For voltage source %s, the value at row %d has been added \
by %E.\n", node_name_index(node_name, "0")+1+i, p->name, 
node_name_index(node_name, "0")+1+i, p->value);
}
i++;
}

if(*p->name=='E' || *p->name=='e')
i++;

if(*p->name=='H' || *p->name=='h')
i++;

p=p->next;
}
S[node_name_index(node_name, "0")]=0;
fprintf(fp1, "S[%d]: This corresponds to ground node equation.\n", 
node_name_index(node_name, "0"));
fprintf(fp1, "\n");
free(p);
}

//function to merge matrices A and B into M
void full_matrix_M(complex **A, complex **B, complex **M, int num_nodes, List *list)
{
int i, j, n_vs=num_voltage_source(list);
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<(num_nodes+n_vs); j++)
{
if(j<num_nodes)
M[i][j]=A[i][j];
else
M[i][j]=B[i][j-num_nodes];
}
}
}

//function to merge matrices V and C into X 
void full_matrix_X(char **v, char **c, char **x, int num_nodes, List *list)
{
int i, j, n_vs=num_voltage_source(list);
//adding entries to matrix x
for(i=0; i<num_nodes+n_vs; i++)
{
if(i<num_nodes)
strcpy(x[i], v[i]);
else
strcpy(x[i], c[i-num_nodes]);
}
}

//function to merge matrices M and S into G 
void augmented_matrix(complex **M, complex *S, complex **G, int num_nodes, List *list)
{
int i, j, n_vs=num_voltage_source(list);
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<(num_nodes+n_vs+1); j++)
{
if(j==(num_nodes+n_vs))
G[i][j]=S[i];
else
G[i][j]=M[i][j];
}
}
}

void swap_largest_element(complex **G, int num_nodes, int n, List *list, FILE *fp1)
{
int i, j, n_vs=num_voltage_source(list);
double max=cabs(G[n][n]);
complex *max_row=(complex *)malloc(sizeof(complex)*(num_nodes+n_vs+1));

for(i=n; i<num_nodes+n_vs; i++)
{
if(max<cabs(G[i][n]))
max=cabs(G[i][n]);
}

for(i=n; i<num_nodes+n_vs; i++)
{
if(max==cabs(G[i][n]))
{
max_row=G[i];
G[i]=G[n];
G[n]=max_row;
fprintf(fp1, "Max element is G[%d][%d].\n", i, n);
fprintf(fp1, "Row %d has been swapped with row %d.\n", i, n); 
if(G[n][n]==0)
{
printf("Singular matrix. Circuit needs to be rebuilt again.\n\n");
fprintf(fp1, "Singular matrix. Circuit needs to be rebuilt again.\n\n");
exit(8);
}
break;
}
}
fprintf(fp1, "The matrix G after swap in column %d is:\n", n);
for(i=0; i<(num_nodes+n_vs+1); i++)
{
fprintf(fp1, "Col %d:%s", i, "                                    ");
}
fprintf(fp1, "\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<(num_nodes+n_vs+1); j++)
{
fprintf(fp1, "Row %d: (%15.7E, %15.7E) ", i, creal(G[i][j]), cimag(G[i][j]));
}
fprintf(fp1, "\n");
}
fprintf(fp1, "\n");
}

void row_transformation(complex **G, int num_nodes, int n, List *list, FILE *fp1)
{
int i, j, n_vs=num_voltage_source(list);
complex z=0, z1=0;
for(i=n+1; i<num_nodes+n_vs; i++)
{
//fprintf(fp1, "In row %d:\n", i);
z=G[i][n];
for(j=n; j<num_nodes+n_vs+1; j++)
{
z1=G[n][j]/G[n][n];
//fprintf(fp1, "z=(%E, %E)\n", creal(z), cimag(z));
//fprintf(fp1, "Previously, G[%d][%d]=(%E, %E)\n", i, j, creal(G[i][j]), cimag(G[i][j]));
G[i][j]-=z*z1;
//fprintf(fp1, "G[%d][%d]-=(z/G[%d][%d])*G[%d][%d] :\n", i, j, n, n, n, j);
//fprintf(fp1, "value ((%E, %E)/(%E, %E))*(%E, %E) = (%E, %E)\n", creal(z), 
//cimag(z), creal(G[n][n]), cimag(G[n][n]), creal(G[n][j]), cimag(G[n][j]), 
//creal((z*G[n][j])/G[n][n]), cimag((z*G[n][j])/G[n][n]));       
//fprintf(fp1, "Now, G[%d][%d]=(%E, %E)\n\n", i, j, creal(G[i][j]), cimag(G[i][j]));
}
}
//fprintf(fp1, "\n");
fprintf(fp1, "The matrix G after row operations in column %d is:\n", n);
for(i=0; i<(num_nodes+n_vs+1); i++)
{
fprintf(fp1, "Col %d:%s", i, "                                    ");
}
fprintf(fp1, "\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<(num_nodes+n_vs+1); j++)
{
fprintf(fp1, "Row %d: (%15.7E, %15.7E) ", i, creal(G[i][j]), cimag(G[i][j]));
}
fprintf(fp1, "\n");           
}
fprintf(fp1, "\n");
}

void solve_echelon_matrix(int num_nodes, List *list, complex *H, complex **G, FILE *fp1)
{
int i, j, n_vs=num_voltage_source(list);
complex z=0, z1=0, z2=0;
for(i=0; i<num_nodes+n_vs; i++)
{
H[i]=0;
}
for(i=num_nodes+n_vs-1; i>=0; i--)
{
H[i]+=G[i][num_nodes+n_vs]/G[i][i];
for(j=num_nodes+n_vs-1; j>i; j--)
{
z1=G[i][j]/G[i][i];
z2=z1*H[j];
H[i]-=z2;
z1=0;
z2=0;
}
}
}

void print_matrix_A(int num_nodes, int n_vs, complex **A, FILE *fp1)
{
int i, j;
fprintf(fp1, "Matrix A has dimensions %d*%d.\n", num_nodes+n_vs, num_nodes);          
fprintf(fp1, "The matrix A is:\n");
for(i=0; i<num_nodes; i++)
{
fprintf(fp1, "Col %d:%s", i, "                                    ");
}
fprintf(fp1, "\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<num_nodes; j++)
{
fprintf(fp1, "Row %d: (%15.7E, %15.7E) ", i, creal(A[i][j]), cimag(A[i][j]));
}
fprintf(fp1, "\n");           
}
fprintf(fp1, "\n");
}

void print_matrix_B(int num_nodes, int n_vs, complex **B, FILE *fp1)
{
int i, j;
fprintf(fp1, "Matrix B has dimensions %d*%d.\n", num_nodes+n_vs, n_vs);        
fprintf(fp1, "The matrix B is:\n");
for(i=0; i<n_vs; i++)
{
fprintf(fp1, "Col %d:%s", i, "                                    ");
}
fprintf(fp1, "\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<n_vs; j++)
{
fprintf(fp1, "Row %d: (%15.7E, %15.7E) ", i, creal(B[i][j]), cimag(B[i][j]));
}
fprintf(fp1, "\n");           
}
fprintf(fp1, "\n");
}

void print_matrix_S(int num_nodes, int n_vs, complex *S, FILE *fp1)
{
int i, j;
fprintf(fp1, "Matrix S has dimensions %d*%d.\n", num_nodes+n_vs, 1);              
fprintf(fp1, "The source vector S is:\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
fprintf(fp1, "%13.5E\n", creal(S[i]));
}
fprintf(fp1, "\n");
}

void print_matrix_M(int num_nodes, int n_vs, complex **M, FILE *fp1)
{
int i, j;
fprintf(fp1, "Matrix M has dimensions %d*%d.\n", num_nodes+n_vs, num_nodes+n_vs);          
fprintf(fp1, "The matrix M is:\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
fprintf(fp1, "Col %d:%s", i, "                                    ");
}
fprintf(fp1, "\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<(num_nodes+n_vs); j++)
{
fprintf(fp1, "Row %d: (%15.7E, %15.7E) ", i, creal(M[i][j]), cimag(M[i][j]));
}
fprintf(fp1, "\n");           
}
fprintf(fp1, "\n");
}

void print_matrix_X(int num_nodes, int n_vs, char **x, FILE *fp1)
{
int i;
fprintf(fp1, "Matrix X has dimensions %d*%d.\n", num_nodes+n_vs, 1);
fprintf(fp1, "The Vector X bar is:\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
fprintf(fp1, "%s\n", x[i]);
}
fprintf(fp1, "\n");
}

void print_matrix_G(int num_nodes, int n_vs, complex **G, int n, FILE *fp1)
{
int i, j;
if(n==0)
fprintf(fp1, "Matrix G before elementary transformations is:\n");
if(n==1)
fprintf(fp1, "Matrix G in echelon form:\n");
for(i=0; i<(num_nodes+n_vs+1); i++)
{
fprintf(fp1, "Col %d:%s", i, "                                    ");
}
fprintf(fp1, "\n");
for(i=0; i<(num_nodes+n_vs); i++)
{
for(j=0; j<(num_nodes+n_vs+1); j++)
{
fprintf(fp1, "Row %d: (%15.7E, %15.7E) ", i, creal(G[i][j]), cimag(G[i][j]));
}
fprintf(fp1, "\n");           
}
fprintf(fp1, "\n");
}

void deallocate(List *list, Node *p, int n, FILE *fp1)
{
int x=n;
while(n>1)
{
p=(p->next);
n-=1;
}
free(p->next);
if(x>1)
fprintf(fp1, "The memory of node at position %d has been freed.\n", x);
if(x-1 > 0)
deallocate(list, list->head, (x-1), fp1);
}

void memory_allocation_1(char **M, int r)
{
int i, j;
for(i=0; i<r; i++)
{
M[i]=(char *)malloc(sizeof(char)*STRLEN);
strcpy(M[i], "0");
}
}

void memory_allocation_2(complex **M, int r, int c)
{
int i, j;
for(i=0; i<r; i++)
{
M[i]=(complex *)malloc(sizeof(complex)*c);
for(j=0; j<c; j++)
{
M[i][j]=0;
}
}
}

void check_voltage_source_analysis(char *source, List *list, FILE *fp1)
{
int err=1;
Node *p=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{
if((*p->name=='V' || *p->name=='v'))
{
if(strcmp(p->name, source)==0)
{
err=0;
break;
}
}
p=p->next;
}
if(err==1)
{
fprintf(fp1, "Given source name for analysis has not been \
defined in the control file.\n");
fprintf(fp1, "Program will terminate.\n\n");
printf("Given source name for analysis has not been defined \
in the control file.\n");
printf("Program will terminate.\n\n");
exit(8);
}
free(p);
}

void create_sources_for_L_and_C(List *list, FILE *fp1)
{
Node *p=(Node *)malloc(sizeof(Node));
Node *new=(Node *)malloc(sizeof(Node));
new->name=(char *)malloc(sizeof(char)*STRLEN);
char *str=(char *)malloc(sizeof(char)*STRLEN);
new->n1=(char *)malloc(sizeof(char)*STRLEN);
new->n2=(char *)malloc(sizeof(char)*STRLEN);
new->n3=new->n4=new->depname=NULL;
p=list->head;
while(p!=NULL)
{
if(*p->name=='C' ||*p->name=='c')
{
strcpy(str, "I_");
str=strcat(str, p->name);
strcpy(new->name, str);
strcpy(new->n1, p->n1);
strcpy(new->n2, p->n2);
new->value=0;
add_node(list, new);
printf("A zero current source %s has been added in place of \
capacitor %s.\n\n", new->name, p->name);
fprintf(fp1, "A zero current source %s has been added in place \
of capacitor %s.\n\n", new->name, p->name);
}
if(*p->name=='L' ||*p->name=='l')
{
strcpy(str, "V_");
str=strcat(str, p->name);
strcpy(new->name, str);
strcpy(new->n1, p->n1);
strcpy(new->n2, p->n2);
new->value=0;
add_node(list, new);
printf("A zero voltage source %s has been added in place of \
inductor %s.\n\n", new->name, p->name);
fprintf(fp1, "A zero voltage source %s has been added in place \
of inductor %s.\n\n", new->name, p->name);
}
p=p->next;
}
}

void print_solution(complex *H, char **x, char **node_name, int n, FILE *fp1, int num)
{
int i, a, n1, n2;
char c;
if(num==1)
{
for(i=0; i<n; i++)
{
a=sscanf(x[i], "%c[%d][%d]", &c, &n1, &n2);
if(a==2)
fprintf(fp1, "%c[%s]=(%E, %E)\n", c, *(node_name+n1), creal(H[i]), cimag(H[i]));
if(a==3)
fprintf(fp1, "%c[%s][%s]=(%E, %E)\n", c, *(node_name+n1), 
*(node_name+n2), creal(H[i]), cimag(H[i]));
}
fprintf(fp1, "\n");
}
if(num==0)
{
for(i=0; i<n; i++)
{
a=sscanf(x[i], "%c[%d][%d]", &c, &n1, &n2);
if(a==2)
printf("%c[%s]=(%E, %E)\n", c, *(node_name+n1), creal(H[i]), cimag(H[i]));
if(a==3)
printf("%c[%s][%s]=(%E, %E)\n", c, *(node_name+n1), 
*(node_name+n2), creal(H[i]), cimag(H[i]));
}
printf("\n");
}
if(num==2)
{
for(i=0; i<n; i++)
{
a=sscanf(x[i], "%c[%d][%d]", &c, &n1, &n2);
if(a==2)
{
fprintf(fp1, "%c[%s]=(%E, %E)\n", c, *(node_name+n1), creal(H[i]), cimag(H[i]));
printf("%c[%s]=(%E, %E)\n", c, *(node_name+n1), creal(H[i]), cimag(H[i]));
}
if(a==3)
{
fprintf(fp1, "%c[%s][%s]=(%E, %E)\n", c, *(node_name+n1), 
*(node_name+n2), creal(H[i]), cimag(H[i]));
printf("%c[%s][%s]=(%E, %E)\n", c, *(node_name+n1), 
*(node_name+n2), creal(H[i]), cimag(H[i]));
}
}
fprintf(fp1, "\n");
printf("\n");
}
if(num==3)
{
for(i=0; i<n; i++)
{
a=sscanf(x[i], "%c[%d][%d]", &c, &n1, &n2);
if(a==2)
fprintf(fp1, "           %c[%s]", c, *(node_name+n1));
if(a==3)
fprintf(fp1, "        %c[%s][%s]", c, *(node_name+n1), *(node_name+n2));
}
fprintf(fp1, "\n");
}
}

void print_solutions_to_file(complex *H, char **node_name, int n, FILE *fp1, 
char *analysis, double value, Node *p)
{
int i, a, n1, n2;
char c;
if(strcmp(analysis, "dc")==0)
{
fprintf(fp1, "%15.7E ", value);
for(i=0; i<n; i++)
{
fprintf(fp1, "%15.7E ", creal(H[i]));
}
fprintf(fp1, "\n");
}

if(strcmp(analysis, "ac")==0)
{
fprintf(fp1, "%15.7E ", value);
fprintf(fp1, "%15.7E ", 
cabs(H[node_name_index(node_name, p->n1)]-H[node_name_index(node_name, p->n2)]));
fprintf(fp1, "%15.7E ", 
carg(H[node_name_index(node_name, p->n1)]-H[node_name_index(node_name, p->n2)]));
fprintf(fp1, "\n");
}
}

Node *check_analysis_branch(char *source, List *list, FILE *fp1)
{
int err=1;
Node *p=(Node *)malloc(sizeof(Node));
Node *new=(Node *)malloc(sizeof(Node));
p=list->head;
while(p!=NULL)
{
if(strcmp(p->name, source)==0)
{
err=0;
new=p;
}
p=p->next;
}
if(err==1)
{
printf("Branch definition not given for ac analysis.\n");
printf("Program will terminate.\n");
fprintf(fp1, "Branch definition not given for ac analysis.\n");
fprintf(fp1, "Program will terminate.\n");
exit(8);
return NULL;
}
else
return new;
}

