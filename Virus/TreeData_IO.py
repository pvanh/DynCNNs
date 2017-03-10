import gl
import pycparser
from Graph import Graph, GVertex

gl.reConstruct= False # reconstruct For, While, DoWhile
gl.ignoreDecl = False # Ignore declaration branches
text ='''
int main()
{
int b = 2;
if (b>2)
    printf("greater than 2");
}
'''

# text ='''
# void main()
# {
# 	int f(int x,int m);
# 	int k,i,j,n,sum=0;
# 	scanf("%d",&n);
# 	for(i=1;i<=n;i++)
# 	{
# 		scanf("%d",&k);
# 		for(j=2;j<=k;j++)
# 		{
# 			if(k%j==0)
# 			{
# 				sum+=f(k,j);
# 			}
# 		}
# 		printf("%d",sum);
# 		sum=0;
# 	}
# }
#
# int f(int x,int m)
# {
# 	int i,sum=0;
# 	if(m==x)
# 		sum=1;
# 	else
# 	{
# 		x=x/m;
# 		for(i=m;i<=x;i++)
# 		{
# 			if(x%i==0)
# 			{
# 				sum+=f(x,i);
# 			}
# 		}
# 	}
# 	return sum;
# }
# '''
def tree2Graph(root):
    vertexes ={}
    edges =[]
    tokdict ={}
    vertex_dict={}
    traverseTree(node= root, vertexes=vertexes, edges=edges,
                     tokdict=tokdict, vertex_dict=vertex_dict, parent_name = '')
    return Graph(vertexes, edges)
def traverseTree( node, vertexes, edges, tokdict, vertex_dict, parent_name=''):
    node_name = node.__class__.__name__
    # check and add the token to token dictionary if not exist
    if node_name in tokdict:
        tokdict[node_name]+=1
    else:
        tokdict[node_name] = 0
    # add current node to vertexes
    v = GVertex(id=0, name=node_name + '_' + str(tokdict[node_name]), token= node_name, toktype='ASM', content='')
    v.id = len(vertexes)
    vertexes[v.id] = v
    vertex_dict[v.name] = len(vertex_dict)
    # add edge from parent ---> current node
    if parent_name !='': # not root node
        edges.append((vertex_dict[parent_name], v.id))
    for (child_name, child) in node.children():
        traverseTree(node= child, vertexes=vertexes, edges=edges,
                     tokdict=tokdict, vertex_dict=vertex_dict, parent_name = v.name)


parser = pycparser.c_parser.CParser()
ast = parser.parse(text=text)  # Parse code to AST
if gl.reConstruct:  # reconstruct braches of For, While, DoWhile
    ast.reConstruct()
print 'AST:'
ast.show()
g =tree2Graph(ast)
g.show()
