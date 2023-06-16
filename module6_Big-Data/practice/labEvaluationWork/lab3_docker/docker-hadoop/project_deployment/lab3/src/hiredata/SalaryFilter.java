package hiredata;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class SalaryFilter {

  public static class SalaryMapper extends Mapper<LongWritable, Text, Text, Text> {
    private final Text outKey = new Text();
    private final Text outValue = new Text();
    private String cityToFilter;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      cityToFilter = conf.get("cityToFilter");
    }

    @Override
    protected void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException {
      String line = value.toString();
      String[] fields = line.split(",");
      String city = fields[1];
      if (city.equals(cityToFilter)) {
        String record = fields[0] + "," + fields[1] + "," + fields[2] + "," + fields[3];
        outKey.set(city);
        outValue.set(record);
        context.write(outKey, outValue);
      }
    }
  }

  public static class SalaryReducer extends Reducer<Text, Text, Text, Text> {
    private final Text result = new Text();

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context)
        throws IOException, InterruptedException {
      for (Text value : values) {
        context.write(key, value);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      System.err.println("Usage: SalaryFilter <inputPath> <outputPath> <city>");
      System.exit(1);
    }

    String inputPath = args[0];
    String outputPath = args[1];
    String cityToFilter = args[2];

    Configuration conf = new Configuration();
    conf.set("cityToFilter", cityToFilter);

    Job job = Job.getInstance(conf, "Salary Filter");
    job.setJarByClass(SalaryFilter.class);
    job.setMapperClass(SalaryMapper.class);
    job.setReducerClass(SalaryReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(inputPath));
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
