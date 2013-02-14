name := "Clustering Leen"

version := "1.0"

scalacOptions ++= Seq("-deprecation", "-unchecked")

libraryDependencies += "com.nicta" %% "scoobi" % "0.6.1-cdh4"

scalacOptions ++= Seq("-Ydependent-method-types")

resolvers ++= Seq(
    "nicta's avro" at "http://nicta.github.com/scoobi/releases",
    "cloudera" at "https://repository.cloudera.com/content/repositories/releases",
    "Sonatype-snapshots" at "http://oss.sonatype.org/content/repositories/snapshots")

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.2-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
            "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
            "Sonatype tools" at "https://oss.sonatype.org/content/groups/scala-tools/"
            )

libraryDependencies += "org.apache.avro" % "avro" % "1.6.3"

mainClass in (Compile, run) := Some("main.scala.KMapRed.Main")
