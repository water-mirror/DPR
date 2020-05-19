import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class java_solverSISR_revised {

    public static void main(String[] args) throws IOException {
//         String data_root = "tmp_datas/";
//         String state_path = "states.txt";
//         String distMat_path = "distMat.txt";
//         String n_steps_str = "1000";
//         String n_ruins_str = "20";

        String data_root = args[0];
        String state_path = args[1];
        String distMat_path = args[2];
        String n_steps_str = args[3];
        String n_ruins_str = args[4];
        String n_anchors_str = args[5];

        int n_steps = Integer.parseInt(n_steps_str);
        int n_ruins = Integer.parseInt(n_ruins_str);
        int n_anchors = Integer.parseInt(n_anchors_str);

        StringBuilder result = new StringBuilder();

        String content = new String(Files.readAllBytes(Paths.get(state_path)), "UTF-8");
        String[] lines = content.split("\n");
        ArrayList<Integer> batch_caps = new ArrayList<>();
        ArrayList<ArrayList<int[]>> batch_routes = new ArrayList<>();
        for (String line : lines) {
            String[] line_split = line.split(":");

            ArrayList<int[]> routes = new ArrayList<>();
            for (String route_str: line_split[2].split(";")) {
                String[] route_arr = route_str.split(",");
                int[] route = new int[route_arr.length];
                for (int i = 0 ; i<route_arr.length ; i++) { route[i] = Integer.parseInt(route_arr[i]); }
                routes.add(route);
            }

            batch_caps.add(Integer.parseInt(line_split[0]));
            batch_routes.add(routes);
        }

        ArrayList<double[][]> datas = new ArrayList<>();
        for (int i = 0 ; i<lines.length ; i++) {
            datas.add(Util_SimpleQuestionReader(data_root+"data_"+Integer.toString(i)+".txt"));
        }

        ArrayList<double[][]> distance_matrices = new ArrayList<>();
        content = new String(Files.readAllBytes(Paths.get(distMat_path)), "UTF-8");
        lines = content.split("\n");
        for (String line : lines) {
            distance_matrices.add(Util_SimpleDistMatReader(line));
        }

        for (int i = 0 ; i<datas.size() ; i++) {
            ArrayList<int[]> new_routes = solve(datas.get(i),
                    batch_caps.get(i),
                    distance_matrices.get(i),
                    n_ruins,
                    n_anchors,
                    n_steps,
                    batch_routes.get(i));
            double new_distance = getRoutesDistance(new_routes, distance_matrices.get(i));
            result.append(new_distance+":"+routes2str(new_routes)+"\n");
        }
        System.out.println(result.toString());
    }

    public static double[][] Util_SimpleDistMatReader(String DistMat) {
        String[] lines = DistMat.split(";");
        double[][] distance_matrix = new double[lines.length][lines.length];
        for (int i = 0 ; i<lines.length ; i++) {
            String[] dists_str = lines[i].split(",");
            for (int j = 0 ; j<dists_str.length ; j++) {
                distance_matrix[i][j] = Double.parseDouble(dists_str[j]);
            }
        }
        return distance_matrix;
    }

    public static double[][] Util_SimpleQuestionReader(String path) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(path)), "UTF-8");
        String[] lines = content.split("\n");
        double[][] data = new double[lines.length][6];
        for (int i = 0 ; i<lines.length ; i++) {
            String[] d = lines[i].split("\\s+");
            for (int j = 0 ; j<d.length ; j++) {
                data[i][j] = Double.parseDouble(d[j]);
            }
        }
        return data;
    }

    public static String routes2str(ArrayList<int[]> routes) {
        StringBuilder sb = new StringBuilder();
        for (int[] r : routes) {
            for (int i : r) {
                sb.append(i);
                sb.append(",");
            }
            sb.deleteCharAt(sb.length()-1);
            sb.append(";");
        }
        sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }

    // 0 is not included in the route but will be calculated.
    public static double getRouteDistance(int[] route, double[][] distance_matrix) {
        int last_node = 0;
        double distance = 0.0;
        for (int c : route) {
            distance += distance_matrix[last_node][c];
            last_node = c;
        }
        distance += distance_matrix[last_node][0];
        return distance;
    }

    // 0 is not included in routes but will be calculated.
    public static double getRoutesDistance(ArrayList<int[]> routes, double[][] distance_matrix) {
        double total_distance = 0.0;
        for (int[] r : routes) {
            total_distance += getRouteDistance(r, distance_matrix);
        }
        return total_distance;
    }

    private static ArrayList<int[]> ruin_routeSummary(ArrayList<int[]> lastRoute, ArrayList<Integer> absents) {
        ArrayList<int[]> absentRoute = new ArrayList<int[]>();
        for (int i = 0 ; i<lastRoute.size() ; i++) {
            ArrayList<Integer> r = new ArrayList<>();
            for (int c : lastRoute.get(i)) {
                if (!absents.contains(c)) r.add(c);
            }
            if (r.size()>0) {
                absentRoute.add(r.stream().mapToInt(j->j).toArray());
            }
        }
        return absentRoute;
    }

    private static int[] insertNode(int[] old_r, int pos, int c) {
        int[] new_r = new int[old_r.length+1];
        for (int i = 0 ; i<new_r.length ; i++) {
            if (i<pos) {
                new_r[i] = old_r[i];
            } else if (i>pos) {
                new_r[i] = old_r[i-1];
            } else {
                new_r[i] = c;
            }
        }
        return new_r;
    }

    private static boolean checkValid(double[][] data, double[][] distance_matrix, int[] r, int c) {
        double time_current = 0;
        int curr_node = 0;
        for (int i = 0 ; i<(r.length+1) ; i++) {
            int next_node = i==r.length?0:r[i];
            time_current+=distance_matrix[curr_node][next_node];
            time_current=Math.max(data[next_node][3], time_current);
            if (time_current<=data[next_node][4]) {
                time_current+=data[next_node][5];
            } else {
                return false;
            }
            curr_node = i==r.length?0:r[i];
        }
        return true;
    }

    private static ArrayList<double[]> getValid(double[][] data, double[][] distance_matrix, int[] r, int c) {
        ArrayList<double[]> valids = new ArrayList<>();
        double dist = getRouteDistance(r, distance_matrix);
        double tmp_time = 0;
        int curr_node = 0;
        for (int i = 0 ; i<(r.length+1) ; i++) {
            int next_node = i==r.length?0:r[i];
            tmp_time = Math.max(tmp_time, data[curr_node][3]);
            tmp_time+=data[curr_node][5];
            if (tmp_time+distance_matrix[curr_node][c]>data[c][4]) break;
            int[] new_r = insertNode(r, i, c);
            if (checkValid(data, distance_matrix, new_r, c)) {
                double new_dist = getRouteDistance(new_r, distance_matrix);
                valids.add(new double[]{i, new_dist-dist});
            }
            tmp_time+=distance_matrix[curr_node][next_node];
            curr_node = i==r.length?0:r[i];
        }
        return valids;
    }

    private static ArrayList<int[]> route_add(ArrayList<int[]> absent_route, int c, double[] adding_pos) {
        if (adding_pos[0]==-1) {
            absent_route.add(new int[]{c});
            return absent_route;
        }
        int[] new_r = insertNode(absent_route.get((int)adding_pos[0]), (int)adding_pos[1], c);
        absent_route.set((int)adding_pos[0], new_r);
        return absent_route;
    }

    private static ArrayList<int[]> recreate(double[][] data, double capcity, double[][] distance_matrix,
                                             ArrayList<int[]> absent_route, ArrayList<Integer> absents, int lastLength) {
        Collections.shuffle(absents);
        ArrayList<Integer> newAbsents = new ArrayList<>();
        ArrayList<Integer> toKeep = new ArrayList<>();
        ArrayList<int[]> current_route = absent_route;
        for (int i = 0 ; i<absents.size() ; i++) {
            int c = absents.get(i);
            ArrayList<double[]> probablePlace = new ArrayList<>();
            for (int ir = 0 ; ir<absent_route.size() ; ir++) {
                int[] r = absent_route.get(ir);
                double dmd_sum = 0;
                for (int _tmp_node : r) dmd_sum+=data[_tmp_node][2];
                if ((dmd_sum+data[c][2])>capcity) continue;
                // all possible int values can round-trip to a double safely.
                ArrayList<double[]> valids = getValid(data, distance_matrix, r, c);
                for (double[] v : valids) probablePlace.add(new double[]{ir,v[0],v[1]});
            }
            double[] adding_pos = new double[]{-1,-1,-1};
            if (probablePlace.size()>0) {
                Collections.sort(probablePlace, new Comparator<double[]>() {
                    public int compare(double[] content0, double[] content1) {
                        if (content0[2]<content1[2]) {
                            return -1;
                        } else if (content0[2]>content1[2]) {
                            return 1;
                        }
                        return 0;
                    }
                });
                adding_pos = probablePlace.get(0);
            } else if (lastLength>0 && lastLength<=current_route.size()) {
                toKeep.add(i);
                continue;
            }
            current_route = route_add(current_route, c, adding_pos);
        }
        for (int i : toKeep) newAbsents.add(absents.get(i));
        absents.clear();
        for (int i : newAbsents) absents.add(i);
        return current_route;
    }

    private static int[][] buildNeighbours(double[][] distance_matrix) {
        int[][] neighbours = new int[distance_matrix.length][distance_matrix.length];
        for (int i = 0 ; i<distance_matrix.length ; i++) {
            java_Util_ArrayIndexComparator comparator = new java_Util_ArrayIndexComparator(distance_matrix[i], false);
            Integer[] indexes = comparator.createIndexArray();
            Arrays.sort(indexes, comparator);
            neighbours[i] = Arrays.stream(indexes).mapToInt(Integer::intValue).toArray();
        }
        return neighbours;
    }

    private static int find_t(ArrayList<int[]> lastRoute, int c) {
        int index = -1;
        for (int i = 0 ; i<lastRoute.size() ; i++) {
            if (Arrays.stream(lastRoute.get(i)).anyMatch(j -> j == c)) {
                index = i;
                break;
            }
        }
        return index;
    }

    private static int get_route_index(int[] route, int c) {
        for (int i = 0 ; i<route.length ; i++) {
            if (route[i]==c) return i;
        }
        return -1;
    }

    private static ArrayList<Integer> remove_nodes(int[] route, int c, int num) {
        ArrayList<Integer> nodes = new ArrayList<>();
        int i = get_route_index(route, c);
        nodes.add(route[i]);
        int j = i+1;
        int k = i-1;
        num = Math.min(num, route.length);
        while(nodes.size()<num) {
            if (j>=route.length) j = 0;
            if (k<0) k = route.length-1;
            nodes.add(route[j]);
            if (nodes.size()<num) nodes.add(route[k]);
            j++;
            k--;
        }
        return nodes;
    }

    public static ArrayList<Integer> find_ruins(int n_ruins, int anchor, double coefficient, int[][] neighbour,
                                                ArrayList<int[]> routes, ArrayList<Integer> ruins) {
        ArrayList<Integer> ruin_nodes = new ArrayList<>();
        coefficient = Math.max(coefficient, 1e-8);
        for (int c : neighbour[anchor]) {
            if (!ruin_nodes.contains(c) && c!=0) {
                int t = find_t(routes, c);
                int num = (int)(Math.ceil(routes.get(t).length * coefficient));
                ArrayList<Integer> newly_removed = remove_nodes(routes.get(t), c, num);
                for (int ruin : newly_removed) {
                    if (ruin_nodes.size()<n_ruins && !ruins.contains(ruin)) {
                        ruin_nodes.add(ruin);
                    } else {
                        break;
                    }
                }
                if (ruin_nodes.size()>=n_ruins) break;
            }
        }
        return ruin_nodes;
    }

    public static ArrayList<int[]> solve(double[][] data, double cap, double[][] distance_matrix,
                                         int n_ruins, int n_anchors, int n_steps, ArrayList<int[]> init_route) {
        double init_T = 100.0;
        double final_T = 1.0;
        double alpha_T = Math.pow((final_T/init_T), (1.0/n_steps));
        double temperature = init_T;

        int[][] neighbours = buildNeighbours(distance_matrix);

        ArrayList<int[]> curr_route = init_route;
        ArrayList<int[]> best_route = init_route;
        double curr_dist = getRoutesDistance(init_route, distance_matrix);
        double best_dist = curr_dist;
        Random random = new Random();
        for (int i_iter = 0 ; i_iter<n_steps ; i_iter++) {
            ArrayList<Integer> ruins = new ArrayList<Integer>();
            for (int i_anchor = 0 ; i_anchor<n_anchors ; i_anchor++) {
                int anchor = random.nextInt(data.length);
                double coefficient = Math.random();
                ArrayList<Integer> ruins_i = find_ruins(n_ruins, anchor, coefficient, neighbours, curr_route, ruins);
                for (int i : ruins_i) ruins.add(i);
            }
            
            ArrayList<int[]> absent_route = ruin_routeSummary(curr_route, ruins);
            ArrayList<int[]> new_route = recreate(data, cap, distance_matrix, absent_route, ruins, 0);
            double new_dist = getRoutesDistance(new_route, distance_matrix);

            if (new_dist<(curr_dist-temperature*Math.log(Math.random()))) {
                curr_route = new_route;
                curr_dist = new_dist;
                if (new_dist<best_dist) {
                    best_route = new_route;
                    best_dist = new_dist;
                }
            }
            temperature*=alpha_T;
        }
        return best_route;
    }

}
